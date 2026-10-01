/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/ucx-exchange/ExchangeCompressionWire.h"
#include "velox/experimental/ucx-exchange/FusedForWire.h"
#include "velox/experimental/ucx-exchange/UcxExchangeRegistration.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Driver.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::ucx_exchange {
namespace {

using cudf_velox::CudfConfig;

class CascadedExchangeTest : public testing::Test {
 protected:
  using Config = std::unordered_map<std::string, std::string>;
  struct Packet {
    std::shared_ptr<cudf::packed_columns> data;
    vector_size_t rows;
    int destination;
  };

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    previousFor_ = fusedForDefault();
    previousCascaded_ = cascadedDefault();
    previousExchange_ = CudfConfig::getInstance().exchange;
    setFusedForDefault(false);
    setCascadedDefault(false);
    CudfConfig::getInstance().exchange = true;
    pool_ = memory::memoryManager()->addLeafPool();
    manager_ = UcxOutputQueueManager::getInstanceRef();
  }

  void TearDown() override {
    for (const auto& [id, partitions] : tasks_) {
      for (int destination = 0; destination < partitions; ++destination) {
        manager_->deleteResults(id, destination);
      }
      manager_->removeTask(id);
    }
    setFusedForDefault(previousFor_);
    setCascadedDefault(previousCascaded_);
    CudfConfig::getInstance().exchange = previousExchange_;
  }

  std::unique_ptr<UcxPartitionedOutput>
  start(const Config& config, int partitions, const RowTypePtr& type) {
    taskId_ = "cascaded-consumer-test-" + std::to_string(tasks_.size());
    task_ = createPartitionedOutputTask(
        taskId_, pool_, type, partitions, {}, FOUR_GBYTES, config);
    manager_->initializeTask(
        task_, core::PartitionedOutputNode::Kind::kPartitioned, partitions, 1);
    tasks_.emplace_back(taskId_, partitions);
    driver_ = std::make_shared<exec::DriverCtx>(
        task_, 0, 0, exec::kUngroupedGroupId, 0);
    auto plan = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
        task_->planFragment().planNode);
    return std::make_unique<UcxPartitionedOutput>(
        0, driver_.get(), plan, manager_);
  }

  std::vector<Packet> produce(
      const std::vector<int32_t>& values,
      const Config& config = {},
      int partitions = 1,
      bool emptyLayout = false) {
    const auto type = emptyLayout ? ROW({}, {}) : ROW({"c0"}, {INTEGER()});
    auto output = start(config, partitions, type);
    const auto rows = static_cast<vector_size_t>(values.size());
    std::vector<std::unique_ptr<cudf::column>> columns;
    if (!emptyLayout) {
      auto column = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT32},
          rows,
          cudf::mask_state::UNALLOCATED,
          stream_.view(),
          cudf::get_current_device_resource_ref());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          column->mutable_view().data<int32_t>(),
          values.data(),
          values.size() * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream_.view().get()));
      columns.push_back(std::move(column));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream_.view().get()));
    auto input = std::make_shared<cudf_velox::CudfVector>(
        output->pool(), type, rows, std::move(table), stream_.view());
    output->addInput(std::move(input));
    output->noMoreInput();
    while (!output->isFinished()) {
      output->getOutput();
    }

    std::vector<Packet> packets;
    for (int destination = 0; destination < partitions; ++destination) {
      while (true) {
        Packet packet{nullptr, 0, destination};
        manager_->getData(
            taskId_,
            destination,
            [&packet](
                std::shared_ptr<cudf::packed_columns> data,
                vector_size_t rows,
                std::vector<int64_t>) {
              packet.data = std::move(data);
              packet.rows = rows;
            });
        if (!packet.data) {
          break;
        }
        packets.push_back(std::move(packet));
      }
    }
    return packets;
  }

  PackedTableWithStreamPtr restore(Packet& packet, cuda::stream_ref stream) {
    auto received = detail::restoreReceivedTable(
        std::move(packet.data->metadata),
        std::move(packet.data->gpu_data),
        stream,
        packet.rows);
    packet.data.reset(); // Returned storage must not depend on the wire owner.
    return received;
  }

  static std::vector<int32_t> readColumn(
      const cudf::column_view& column,
      cuda::stream_ref stream) {
    std::vector<int32_t> values(column.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        values.data(),
        column.data<int32_t>(),
        values.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return values;
  }

  void expectCascaded(std::vector<Packet>& packets, vector_size_t totalRows) {
    vector_size_t rows = 0;
    rmm::cuda_stream consumer;
    for (auto& packet : packets) {
      auto decoded = unwrapExchangePayloadMetadata(
          std::make_unique<std::vector<uint8_t>>(*packet.data->metadata));
      ASSERT_EQ(decoded.codec, ExchangePayloadCodec::kCascaded);
      EXPECT_LT(packet.data->gpu_data->size(), decoded.logicalDataSize);
      // resize must change the transmitted size, not just metadata. Retained
      // allocation capacity is not the payload size and is not sent.
      EXPECT_LT(
          packet.data->gpu_data->size(), packet.data->gpu_data->capacity());
      auto received = restore(packet, consumer.view());
      ASSERT_TRUE(received->table);
      EXPECT_FALSE(received->packedTable);
      EXPECT_EQ(received->gpuDataSize(), decoded.logicalDataSize);
      EXPECT_EQ(received->numRows, packet.rows);
      EXPECT_EQ(received->tableView().num_rows(), packet.rows);
      EXPECT_EQ(
          readColumn(received->tableView().column(0), consumer.view()),
          std::vector<int32_t>(packet.rows, 7));
      rows += packet.rows;
    }
    EXPECT_EQ(rows, totalRows);
  }

  rmm::cuda_stream stream_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> manager_;
  std::shared_ptr<exec::Task> task_;
  std::shared_ptr<exec::DriverCtx> driver_;
  std::string taskId_;
  std::vector<std::pair<std::string, int>> tasks_;
  bool previousFor_;
  bool previousCascaded_;
  bool previousExchange_;
};

TEST_F(CascadedExchangeTest, disabledByDefaultRetainsRawPacking) {
  EXPECT_FALSE(cascadedDefault());
  const std::vector<int32_t> values(65536, 7);
  auto packets = produce(values);
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
  auto* allocation = packets[0].data->gpu_data.get();
  auto received = restore(packets[0], stream_.view());
  EXPECT_FALSE(received->table);
  ASSERT_TRUE(received->packedTable);
  EXPECT_EQ(received->packedTable->data.gpu_data.get(), allocation);
  EXPECT_EQ(
      readColumn(received->tableView().column(0), stream_.view()), values);
}

TEST_F(CascadedExchangeTest, singleDestinationUsesCompactOwningPath) {
  auto packets = produce(
      std::vector<int32_t>(65536, 7), {{CudfConfig::kUcxCascaded, "true"}});
  ASSERT_EQ(packets.size(), 1);
  expectCascaded(packets, 65536);
}

TEST_F(CascadedExchangeTest, splitDestinationsUseCompactOwningPath) {
  // Establish that the fixture actually produces both destinations without
  // compression before checking the new codec path.
  auto raw = produce(std::vector<int32_t>(65536, 7), {}, 2);
  ASSERT_EQ(raw.size(), 2);
  for (auto& packet : raw) {
    EXPECT_EQ(packet.rows, 32768);
    auto received = restore(packet, stream_.view());
    ASSERT_TRUE(received->packedTable);
    EXPECT_EQ(
        readColumn(received->tableView().column(0), stream_.view()),
        std::vector<int32_t>(32768, 7));
  }
  auto packets = produce(
      std::vector<int32_t>(65536, 7), {{CudfConfig::kUcxCascaded, "true"}}, 2);
  ASSERT_EQ(packets.size(), 2);
  EXPECT_EQ(packets[0].destination, 0);
  EXPECT_EQ(packets[1].destination, 1);
  EXPECT_EQ(packets[0].rows, 32768);
  EXPECT_EQ(packets[1].rows, 32768);
  expectCascaded(packets, 65536);
}

TEST_F(CascadedExchangeTest, noReductionKeepsOrdinaryPayload) {
  // One scalar occupies a padded raw region smaller than a native Cascaded
  // bitstream header plus its chunk. This exercises the no-gain fallback.
  auto packets = produce({37}, {{CudfConfig::kUcxCascaded, "true"}});
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
  auto received = restore(packets[0], stream_.view());
  ASSERT_TRUE(received->packedTable);
  EXPECT_FALSE(received->table);
  EXPECT_EQ(
      readColumn(received->tableView().column(0), stream_.view()),
      (std::vector<int32_t>{37}));
}

TEST_F(CascadedExchangeTest, emptyLayoutKeepsProducerRowsAndRawStorage) {
  auto packets = produce(
      std::vector<int32_t>(7), {{CudfConfig::kUcxCascaded, "true"}}, 1, true);
  ASSERT_EQ(packets.size(), 1);
  EXPECT_EQ(packets[0].data->gpu_data->size(), 0);
  auto received = restore(packets[0], stream_.view());
  ASSERT_TRUE(received->packedTable);
  EXPECT_FALSE(received->table);
  EXPECT_EQ(received->numRows, 7);
  EXPECT_EQ(received->tableView().num_columns(), 0);
}

TEST_F(CascadedExchangeTest, oldForEnvelopeAndDirectDecoderAreRetained) {
  const std::vector<int32_t> values(65536, 7);
  auto packets = produce(values, {{CudfConfig::kUcxFusedFor, "true"}});
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapFusedForMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  ASSERT_TRUE(metadata.encoded);
  EXPECT_GT(metadata.segmentCount, 0);
  auto received = restore(packets[0], stream_.view());
  ASSERT_TRUE(received->packedTable);
  EXPECT_FALSE(received->table);
  EXPECT_EQ(
      readColumn(received->tableView().column(0), stream_.view()), values);
}

TEST_F(CascadedExchangeTest, rejectsSimultaneousForAndCascaded) {
  EXPECT_THROW(
      start(
          {{CudfConfig::kUcxFusedFor, "true"},
           {CudfConfig::kUcxCascaded, "true"}},
          1,
          ROW({"c0"}, {INTEGER()})),
      VeloxUserError);
}

TEST_F(CascadedExchangeTest, processDefaultSupportsQueryOverride) {
  CudfConfig::getInstance().initialize({{CudfConfig::kUcxCascaded, "true"}});
  EXPECT_TRUE(cascadedDefault());
  auto compressed = produce(std::vector<int32_t>(65536, 7));
  ASSERT_EQ(compressed.size(), 1);
  expectCascaded(compressed, 65536);
  auto raw = produce(
      std::vector<int32_t>(65536, 7), {{CudfConfig::kUcxCascaded, "false"}});
  ASSERT_EQ(raw.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*raw[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
  EXPECT_THROW(
      start({{CudfConfig::kUcxFusedFor, "true"}}, 1, ROW({"c0"}, {INTEGER()})),
      VeloxUserError);
}

TEST_F(CascadedExchangeTest, receiverRejectsTruncatedCompressedPayload) {
  auto packets = produce(
      std::vector<int32_t>(65536, 7), {{CudfConfig::kUcxCascaded, "true"}});
  ASSERT_EQ(packets.size(), 1);
  auto& wire = *packets[0].data->gpu_data;
  ASSERT_GT(wire.size(), 1);
  wire.resize(wire.size() - 1, stream_.view());
  EXPECT_ANY_THROW(restore(packets[0], stream_.view()));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
