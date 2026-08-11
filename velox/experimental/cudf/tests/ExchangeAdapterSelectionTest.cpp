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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Merge.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

namespace facebook::velox::exec::test {
namespace {

using core::TransportKind;

class ExchangeAdapterSelectionTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedEnabled_ = cudf_velox::CudfConfig::getInstance().enabled;
    savedExchange_ = cudf_velox::CudfConfig::getInstance().exchange;
    cudf_velox::CudfConfig::getInstance().enabled = true;
    cudf_velox::CudfConfig::getInstance().exchange = true;
    OutputTransportRegistry::unregisterAll();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::OperatorAdapterRegistry::getInstance().clear();
    OutputTransportRegistry::unregisterAll();
    cudf_velox::CudfConfig::getInstance().enabled = savedEnabled_;
    cudf_velox::CudfConfig::getInstance().exchange = savedExchange_;
    OperatorTestBase::TearDown();
  }

  core::PlanFragment makeExchangePlan() {
    return PlanBuilder()
        .exchange(rowType_, VectorSerde::kindName(VectorSerde::Kind::kPresto))
        .planFragment();
  }

  core::PlanFragment makeMergeExchangePlan() {
    return PlanBuilder()
        .mergeExchange(
            rowType_, {"c0"}, VectorSerde::kindName(VectorSerde::Kind::kPresto))
        .planFragment();
  }

  std::shared_ptr<core::QueryCtx> makeQueryCtx(bool cudfEnabled,
                                               bool ucxExchange) {
    return core::QueryCtx::create(
        nullptr,
        core::QueryConfig{{{std::string(cudf_velox::CudfConfig::kCudfEnabled),
                            cudfEnabled ? "true" : "false"},
                           {std::string(cudf_velox::CudfConfig::kUcxExchange),
                            ucxExchange ? "true" : "false"}}});
  }

  std::shared_ptr<Task> makeTask(std::string taskId,
                                 core::PlanFragment fragment,
                                 std::shared_ptr<core::QueryCtx> queryCtx) {
    return Task::create(std::move(taskId),
                        std::move(fragment),
                        0,
                        std::move(queryCtx),
                        Task::ExecutionMode::kParallel);
  }

  std::shared_ptr<DriverCtx> makeDriverCtx(std::shared_ptr<Task> task) {
    return std::make_shared<DriverCtx>(
        std::move(task), 0, 0, kUngroupedGroupId, 0);
  }

  RowTypePtr rowType_{ROW({"c0", "c1"}, {BIGINT(), VARCHAR()})};

 private:
  bool savedEnabled_{false};
  bool savedExchange_{false};
};

TEST_F(ExchangeAdapterSelectionTest, registersPairedUcxOutputTransport) {
  auto entry =
      OutputTransportRegistry::tryGet(std::string{TransportKind::kUcx});
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<ucx_exchange::UcxOutputQueueManager>(
                entry->manager),
            nullptr);
  EXPECT_TRUE(static_cast<bool>(entry->makeOutputOperator));

  auto fragment =
      PlanBuilder()
          .values({makeRowVector(rowType_, 1)})
          .partitionedOutput({"c0"},
                             4,
                             false,
                             {},
                             VectorSerde::kindName(VectorSerde::Kind::kPresto),
                             std::string{TransportKind::kUcx})
          .planFragment();
  auto node = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
      fragment.planNode);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->transportKind(), TransportKind::kUcx);
}

TEST_F(ExchangeAdapterSelectionTest, recognizesExistingUcxOutputAsGpuBoundary) {
  auto entry =
      OutputTransportRegistry::tryGet(std::string{TransportKind::kUcx});
  ASSERT_NE(entry, nullptr);

  auto fragment =
      PlanBuilder()
          .values({makeRowVector(rowType_, 1)})
          .partitionedOutput({"c0"},
                             4,
                             false,
                             {},
                             VectorSerde::kindName(VectorSerde::Kind::kPresto),
                             std::string{TransportKind::kUcx})
          .planFragment();
  auto node = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
      fragment.planNode);
  ASSERT_NE(node, nullptr);
  auto task = makeTask(
      "existing-ucx-output-adapter",
      std::move(fragment),
      makeQueryCtx(true, true));
  auto driverCtx = makeDriverCtx(task);
  auto op = entry->makeOutputOperator(0, driverCtx.get(), node, false);
  ASSERT_NE(
      dynamic_cast<ucx_exchange::UcxPartitionedOutput*>(op.get()), nullptr);

  auto* adapter =
      cudf_velox::OperatorAdapterRegistry::getInstance().findAdapter(op.get());
  ASSERT_NE(adapter, nullptr);
  EXPECT_TRUE(adapter->canRunOnGPU(op.get(), node, driverCtx.get()));
  EXPECT_TRUE(adapter->acceptsGpuInput());
  EXPECT_FALSE(adapter->producesGpuOutput());
  EXPECT_TRUE(adapter->keepOperator(op.get(), node, driverCtx.get()));
  EXPECT_TRUE(
      adapter->createReplacements(op.get(), node, driverCtx.get(), 0).empty());
}

TEST_F(ExchangeAdapterSelectionTest,
       recognizesInMemoryOutputAsIntentionalCpuBoundary) {
  auto fragment =
      PlanBuilder()
          .values({makeRowVector(rowType_, 1)})
          .partitionedOutput(
              {"c0"},
              4,
              false,
              {},
              VectorSerde::kindName(VectorSerde::Kind::kPresto),
              std::string{TransportKind::kInMemory})
          .planFragment();
  auto node = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
      fragment.planNode);
  ASSERT_NE(node, nullptr);
  auto task = makeTask(
      "in-memory-output-adapter",
      std::move(fragment),
      makeQueryCtx(true, true));
  auto driverCtx = makeDriverCtx(task);
  PartitionedOutput op(
      0,
      driverCtx.get(),
      node,
      false,
      DefaultOutputBufferManager::getInstanceRef());

  auto* adapter =
      cudf_velox::OperatorAdapterRegistry::getInstance().findAdapter(&op);
  ASSERT_NE(adapter, nullptr);
  EXPECT_FALSE(adapter->canRunOnGPU(&op, node, driverCtx.get()));
  EXPECT_FALSE(adapter->acceptsGpuInput());
  EXPECT_FALSE(adapter->producesGpuOutput());
  EXPECT_TRUE(adapter->keepOperator(&op, node, driverCtx.get()));
  EXPECT_TRUE(
      adapter->createReplacements(&op, node, driverCtx.get(), 0).empty());
}

TEST_F(ExchangeAdapterSelectionTest, exchangeHonorsPerQueryFlags) {
  struct Case {
    bool cudfEnabled;
    bool ucxExchange;
    bool expectReplacement;
  };
  const std::vector<Case> cases = {
      {true, true, true},
      {true, false, false},
      {false, true, false},
      {false, false, false},
  };

  auto& registry = cudf_velox::OperatorAdapterRegistry::getInstance();
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& testCase = cases[i];
    SCOPED_TRACE(fmt::format("cudfEnabled={} ucxExchange={}",
                             testCase.cudfEnabled,
                             testCase.ucxExchange));

    auto fragment = makeExchangePlan();
    auto node = fragment.planNode;
    auto task =
        makeTask(fmt::format("exchange-flags-{}", i),
                 std::move(fragment),
                 makeQueryCtx(testCase.cudfEnabled, testCase.ucxExchange));
    auto driverCtx = makeDriverCtx(task);
    Exchange op(0,
                driverCtx.get(),
                std::dynamic_pointer_cast<const core::ExchangeNode>(node),
                nullptr);

    auto* adapter = registry.findAdapter(&op);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->canRunOnGPU(&op, node, driverCtx.get()),
              testCase.expectReplacement);
    EXPECT_EQ(!adapter->keepOperator(&op, node, driverCtx.get()),
              testCase.expectReplacement);
  }
}

TEST_F(ExchangeAdapterSelectionTest, mergeExchangeHonorsPerQueryFlags) {
  struct Case {
    bool cudfEnabled;
    bool ucxExchange;
    bool expectReplacement;
  };
  const std::vector<Case> cases = {
      {true, true, true},
      {true, false, false},
      {false, true, false},
      {false, false, false},
  };

  auto& registry = cudf_velox::OperatorAdapterRegistry::getInstance();
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& testCase = cases[i];
    SCOPED_TRACE(fmt::format("cudfEnabled={} ucxExchange={}",
                             testCase.cudfEnabled,
                             testCase.ucxExchange));

    auto fragment = makeMergeExchangePlan();
    auto node = fragment.planNode;
    auto task =
        makeTask(fmt::format("merge-exchange-flags-{}", i),
                 std::move(fragment),
                 makeQueryCtx(testCase.cudfEnabled, testCase.ucxExchange));
    auto driverCtx = makeDriverCtx(task);
    MergeExchange op(
        0,
        driverCtx.get(),
        std::dynamic_pointer_cast<const core::MergeExchangeNode>(node));

    auto* adapter = registry.findAdapter(&op);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->canRunOnGPU(&op, node, driverCtx.get()),
              testCase.expectReplacement);
    EXPECT_EQ(!adapter->keepOperator(&op, node, driverCtx.get()),
              testCase.expectReplacement);
  }
}

TEST_F(ExchangeAdapterSelectionTest, requiresQueryVisibleUcxTransport) {
  auto fragment = makeExchangePlan();
  auto node = fragment.planNode;
  auto queryCtx = makeQueryCtx(true, true);
  queryCtx->setRegistry(OutputTransportRegistry::kRegistryKey,
                        OutputTransportRegistry::create(nullptr));
  auto task =
      makeTask("exchange-isolated-registry", std::move(fragment), queryCtx);
  auto driverCtx = makeDriverCtx(task);
  Exchange op(0,
              driverCtx.get(),
              std::dynamic_pointer_cast<const core::ExchangeNode>(node),
              nullptr);

  auto* adapter =
      cudf_velox::OperatorAdapterRegistry::getInstance().findAdapter(&op);
  ASSERT_NE(adapter, nullptr);
  EXPECT_FALSE(adapter->canRunOnGPU(&op, node, driverCtx.get()));
  EXPECT_TRUE(adapter->keepOperator(&op, node, driverCtx.get()));
}

TEST_F(ExchangeAdapterSelectionTest, globalExchangeDisableSkipsAdapter) {
  cudf_velox::CudfConfig::getInstance().exchange = false;
  auto fragment = makeExchangePlan();
  auto node = fragment.planNode;
  auto task = makeTask("exchange-global-disabled",
                       std::move(fragment),
                       makeQueryCtx(true, true));
  auto driverCtx = makeDriverCtx(task);
  Exchange op(0,
              driverCtx.get(),
              std::dynamic_pointer_cast<const core::ExchangeNode>(node),
              nullptr);

  EXPECT_EQ(cudf_velox::OperatorAdapterRegistry::getInstance().findAdapter(&op),
            nullptr);
}

} // namespace
} // namespace facebook::velox::exec::test
