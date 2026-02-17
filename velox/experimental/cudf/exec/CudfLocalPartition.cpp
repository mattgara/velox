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

#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/Task.h"

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

std::string toString128(__int128_t value) {
  if (value == 0) {
    return "0";
  }
  bool negative = value < 0;
  __int128_t absValue = negative ? -value : value;
  std::string out;
  while (absValue > 0) {
    int digit = static_cast<int>(absValue % 10);
    out.push_back(static_cast<char>('0' + digit));
    absValue /= 10;
  }
  if (negative) {
    out.push_back('-');
  }
  std::reverse(out.begin(), out.end());
  return out;
}

bool isValidAt(const std::vector<uint8_t>& mask, int64_t index) {
  if (mask.empty()) {
    return true;
  }
  auto const byte = mask[static_cast<size_t>(index) / 8];
  return ((byte >> (index % 8)) & 1) != 0;
}

struct TrackKeyPartitionStats {
  int64_t rows{0};
  int64_t stateRows{0};
  __int128_t sum{0};
  int64_t count{0};
  bool stateValid{false};
};

void trackPartitionKeys(
    cudf::table_view const& tableView,
    std::vector<int64_t> const& keys,
    std::vector<column_index_t> const& keyIndices,
    rmm::cuda_stream_view stream,
    std::string const& taskId,
    std::string const& planNodeId,
    int32_t operatorId,
    uint32_t splitGroupId,
    int32_t partitionIndex,
    const void* vectorPtr) {
  if (keys.empty()) {
    return;
  }
  if (keyIndices.size() != 1) {
    LOG(INFO) << "[HashAggPartitionTrack] skipped reason=multiKey"
              << " task=" << taskId << " plan=" << planNodeId
              << " op=" << operatorId << " split=" << splitGroupId
              << " partition=" << partitionIndex;
    return;
  }
  auto const keyIndex = static_cast<cudf::size_type>(keyIndices[0]);
  if (keyIndex >= tableView.num_columns()) {
    return;
  }
  auto keyCol = tableView.column(keyIndex);
  auto keyType = keyCol.type().id();
  if (keyType != cudf::type_id::INT64 &&
      keyType != cudf::type_id::INT32) {
    return;
  }

  int32_t stateIndex = -1;
  for (cudf::size_type i = 0; i < tableView.num_columns(); ++i) {
    if (i == keyIndex) {
      continue;
    }
    if (tableView.column(i).type().id() == cudf::type_id::STRING) {
      stateIndex = static_cast<int32_t>(i);
      break;
    }
  }

  std::unordered_map<int64_t, TrackKeyPartitionStats> stats;
  stats.reserve(keys.size());
  for (auto key : keys) {
    stats.emplace(key, TrackKeyPartitionStats{});
  }

  int64_t totalRows = tableView.num_rows();
  auto const& cfg = CudfConfig::getInstance();
  if (cfg.debugHashAggEndToEndMaxRows > 0 &&
      totalRows > cfg.debugHashAggEndToEndMaxRows) {
    totalRows = cfg.debugHashAggEndToEndMaxRows;
  }
  int64_t batchRows = cfg.debugHashAggEndToEndBatchRows;
  if (batchRows <= 0) {
    batchRows = totalRows;
  }

  auto keyElementSize = cudf::size_of(keyCol.type());
  auto const* keyBase = static_cast<const uint8_t*>(keyCol.head()) +
      keyCol.offset() * keyElementSize;
  auto keyBitOffset = keyCol.offset();

  std::vector<uint8_t> keyMask;
  if (keyCol.null_count() > 0) {
    auto const maskBytes = static_cast<size_t>(
        cudf::bitmask_allocation_size_bytes(keyCol.offset() + keyCol.size()));
    keyMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        keyMask.data(),
        keyCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      return;
    }
  }

  std::unique_ptr<cudf::column> decodedSum;
  std::unique_ptr<cudf::column> decodedCount;
  std::vector<uint8_t> decodedMask;
  cudf::type_id sumType = cudf::type_id::DECIMAL128;
  int64_t sumElementSize = 0;
  int64_t countElementSize = 0;
  const uint8_t* sumBase = nullptr;
  const uint8_t* countBase = nullptr;
  int64_t sumBitOffset = 0;
  bool hasState = false;
  if (stateIndex >= 0) {
    auto stateCol = tableView.column(stateIndex);
    auto decoded =
        deserializeDecimalSumStateWithCount(stateCol, 0, stream);
    decodedSum = std::move(decoded.sum);
    decodedCount = std::move(decoded.count);
    hasState = true;
    auto sumView = decodedSum->view();
    auto countView = decodedCount->view();
    sumType = sumView.type().id();
    sumElementSize = cudf::size_of(sumView.type());
    countElementSize = cudf::size_of(countView.type());
    sumBase = static_cast<const uint8_t*>(sumView.head()) +
        sumView.offset() * sumElementSize;
    countBase = static_cast<const uint8_t*>(countView.head()) +
        countView.offset() * countElementSize;
    sumBitOffset = sumView.offset();
    if (sumView.null_count() > 0) {
      auto const maskBytes = static_cast<size_t>(
          cudf::bitmask_allocation_size_bytes(sumView.offset() + sumView.size()));
      decodedMask.resize(maskBytes);
      auto const copyStatus = cudaMemcpyAsync(
          decodedMask.data(),
          sumView.null_mask(),
          maskBytes,
          cudaMemcpyDeviceToHost,
          stream.value());
      if (copyStatus != cudaSuccess) {
        return;
      }
    }
  }

  if (!keyMask.empty() || !decodedMask.empty()) {
    auto const syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      return;
    }
  }

  std::vector<uint8_t> keyHost;
  std::vector<uint8_t> sumHost;
  std::vector<uint8_t> countHost;

  for (int64_t start = 0; start < totalRows; start += batchRows) {
    auto const rowsThis = std::min<int64_t>(batchRows, totalRows - start);
    auto const keyBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(keyElementSize);
    keyHost.resize(keyBytes);
    if (hasState) {
      sumHost.resize(static_cast<size_t>(rowsThis) *
                     static_cast<size_t>(sumElementSize));
      countHost.resize(static_cast<size_t>(rowsThis) *
                       static_cast<size_t>(countElementSize));
    }

    auto const* keyPtr = keyBase + start * keyElementSize;
    auto keyStatus = cudaMemcpyAsync(
        keyHost.data(),
        keyPtr,
        keyBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    cudaError_t sumStatus = cudaSuccess;
    cudaError_t countStatus = cudaSuccess;
    if (hasState) {
      auto const* sumPtr = sumBase + start * sumElementSize;
      auto const* countPtr = countBase + start * countElementSize;
      sumStatus = cudaMemcpyAsync(
          sumHost.data(),
          sumPtr,
          static_cast<size_t>(rowsThis) * static_cast<size_t>(sumElementSize),
          cudaMemcpyDeviceToHost,
          stream.value());
      countStatus = cudaMemcpyAsync(
          countHost.data(),
          countPtr,
          static_cast<size_t>(rowsThis) * static_cast<size_t>(countElementSize),
          cudaMemcpyDeviceToHost,
          stream.value());
    }
    if (keyStatus != cudaSuccess || sumStatus != cudaSuccess ||
        countStatus != cudaSuccess) {
      return;
    }
    auto syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      return;
    }

    for (int64_t i = 0; i < rowsThis; ++i) {
      auto const rowIndex = start + i;
      if (!isValidAt(keyMask, keyBitOffset + rowIndex)) {
        continue;
      }
      int64_t key = 0;
      if (keyType == cudf::type_id::INT64) {
        std::memcpy(&key, keyHost.data() + i * keyElementSize, sizeof(int64_t));
      } else {
        int32_t key32 = 0;
        std::memcpy(
            &key32, keyHost.data() + i * keyElementSize, sizeof(int32_t));
        key = key32;
      }
      auto it = stats.find(key);
      if (it == stats.end()) {
        continue;
      }
      auto& entry = it->second;
      entry.rows += 1;
      if (!hasState) {
        continue;
      }
      if (!decodedMask.empty() &&
          !isValidAt(decodedMask, sumBitOffset + rowIndex)) {
        continue;
      }
      entry.stateRows += 1;
      entry.stateValid = true;
      __int128_t sumValue = 0;
      if (sumType == cudf::type_id::DECIMAL64) {
        int64_t value64 = 0;
        std::memcpy(
            &value64,
            sumHost.data() + i * sumElementSize,
            sizeof(int64_t));
        sumValue = static_cast<__int128_t>(value64);
      } else {
        uint64_t lo = 0;
        int64_t hi = 0;
        auto const* ptr = sumHost.data() + i * sumElementSize;
        std::memcpy(&lo, ptr, sizeof(uint64_t));
        std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
        sumValue = (static_cast<__int128_t>(hi) << 64) | lo;
      }
      entry.sum += sumValue;
      int64_t countValue = 0;
      std::memcpy(
          &countValue,
          countHost.data() + i * countElementSize,
          sizeof(int64_t));
      entry.count += countValue;
    }
  }

  for (auto const& key : keys) {
    auto it = stats.find(key);
    if (it == stats.end()) {
      continue;
    }
    auto const& entry = it->second;
    if (entry.rows == 0 && entry.stateRows == 0) {
      continue;
    }
    LOG(INFO) << "[HashAggPartitionTrack] task=" << taskId
              << " plan=" << planNodeId
              << " op=" << operatorId
              << " split=" << splitGroupId
              << " partition=" << partitionIndex
              << " ptr=" << vectorPtr
              << " key=" << key
              << " rows=" << entry.rows
              << " stateRows=" << entry.stateRows
              << " stateCount=" << entry.count
              << " stateSum=" << toString128(entry.sum)
              << " stateValid=" << entry.stateValid
              << " stateIndex=" << stateIndex;
  }
}

} // namespace

bool CudfLocalPartition::shouldReplace(
    const std::shared_ptr<const core::LocalPartitionNode>& planNode) {
  auto* hashFunctionSpec = dynamic_cast<const exec::HashPartitionFunctionSpec*>(
      &planNode->partitionFunctionSpec());
  // Only replace LocalPartition with CudfLocalPartition for hash partitioning.
  // TODO: Round Robin Row-Wise Partitioning can be supported in future.
  return hashFunctionSpec;
}

CudfLocalPartition::CudfLocalPartition(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::LocalPartitionNode>& planNode)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfLocalPartition"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()} {
  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  std::string spec = planNode->partitionFunctionSpec().toString();
  auto* hashFunctionSpec = dynamic_cast<const exec::HashPartitionFunctionSpec*>(
      &planNode->partitionFunctionSpec());

  // Only parse keys if it's a hash function
  if (hashFunctionSpec) {
    // Extract keys between HASH( and )
    size_t start = spec.find("HASH(") + 5;
    size_t end = spec.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
    partitionFunctionType_ = PartitionFunctionType::kHash;
  } else if (
      numPartitions_ > 1 && spec.find("ROUND ROBIN") != std::string::npos) {
    partitionFunctionType_ = PartitionFunctionType::kRoundRobin;
  }
  VELOX_CHECK(
      numPartitions_ == 1 || partitionKeyIndices_.size() > 0 ||
      partitionFunctionType_ == PartitionFunctionType::kRoundRobin);

  // Since we're replacing the LocalPartition with CudfLocalPartition, the
  // number of producers is already set. Adding producer only adds to a counter
  // which we don't have to do again.
  // Normally, this is what we'd have to do:
  // for (auto& queue : queues_) {
  //   queue->addProducer();
  // }
}

void CudfLocalPartition::recordOutputStats(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }
}

void CudfLocalPartition::flushVectorPool() {
  // We reuse the LocalExchangeQueue from the CPU implementation. That impl
  // stores used vectors in a vector pool for the CPU LocalPartition to re-use.
  // CudfLocalPartition does not need it and does not extract it. This results
  // in unnecessary extension of the lifetimes of vectors that were exchanged,
  // resulting in kind of a memory leak.
  // This is a hack to forcefully flush the vector pools.

  for (auto& queue : queues_) {
    queue->getVector();
  }
}

void CudfLocalPartition::addInput(RowVectorPtr input) {
  flushVectorPool();
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  recordOutputStats(input);
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");
  auto stream = cudfVector->stream();

  if (numPartitions_ > 1) {
    auto [partitionedTable, partitionOffsets] = [&]() {
      auto tableView = cudfVector->getTableView();
      // Use cudf hash partitioning
      if (partitionFunctionType_ == PartitionFunctionType::kHash) {
        std::vector<cudf::size_type> partitionKeyIndices;
        for (const auto& idx : partitionKeyIndices_) {
          partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
        }

        return cudf::hash_partition(
            tableView,
            partitionKeyIndices,
            numPartitions_,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream);
      } else if (partitionFunctionType_ == PartitionFunctionType::kRoundRobin) {
        return cudf::round_robin_partition(
            tableView, numPartitions_, counter_, stream);
        counter_ = (counter_ + cudfVector->size()) % numPartitions_;
      }
      VELOX_FAIL("Unsupported partition function");
    }();

    // cuDF partitioning APIs return num_partitions + 1 offsets where:
    // - offsets[i] is the starting row index for partition i
    // - offsets[num_partitions] is the total row count
    VELOX_CHECK(partitionOffsets.size() == numPartitions_ + 1);
    VELOX_CHECK(partitionOffsets[0] == 0);

    // cudf::split expects split points (excluding first 0 and last totalRows).
    // Erase first element (always 0) and last element (total row count).
    partitionOffsets.erase(partitionOffsets.begin());
    partitionOffsets.pop_back();

    auto partitionedTables =
        cudf::split(partitionedTable->view(), partitionOffsets, stream);

    for (int i = 0; i < numPartitions_; ++i) {
      auto partitionData = partitionedTables[i];
      if (partitionData.num_rows() == 0) {
        // Skip empty partitions.
        continue;
      }
      auto cudfPartitionVector = std::make_shared<CudfVector>(
          pool(),
          outputType_,
          partitionData.num_rows(),
          std::make_unique<cudf::table>(partitionData, stream),
          stream);
      if (!CudfConfig::getInstance().debugHashAggTrackKeys.empty()) {
        trackPartitionKeys(
            partitionData,
            CudfConfig::getInstance().debugHashAggTrackKeys,
            partitionKeyIndices_,
            stream,
            taskId(),
            planNodeId(),
            operatorId(),
            splitGroupId(),
            i,
            cudfPartitionVector.get());
      }

      ContinueFuture future;
      // DM: We should investigate if keeping partitionedTables alive and using
      // the table view in partitonedData is more efficient than creating a new
      // table each time. Currently out of scope because it would need a new
      // type of RowVector that can hold a table view and shared_ptr to the
      // table.
      auto blockingReason = queues_[i]->enqueue(
          cudfPartitionVector,
          partitionData.num_rows(),
          &future);
      if (blockingReason != exec::BlockingReason::kNotBlocked) {
        blockingReasons_.push_back(blockingReason);
        futures_.push_back(std::move(future));
      }
    }
  } else {
    // Single partition case.
    ContinueFuture future;
    auto tableView = cudfVector->getTableView();
    auto subVector = std::make_shared<CudfVector>(
        pool(),
        outputType_,
        tableView.num_rows(),
        std::make_unique<cudf::table>(tableView, stream),
        stream);
    if (!CudfConfig::getInstance().debugHashAggTrackKeys.empty()) {
      trackPartitionKeys(
          tableView,
          CudfConfig::getInstance().debugHashAggTrackKeys,
          partitionKeyIndices_,
          stream,
          taskId(),
          planNodeId(),
          operatorId(),
          splitGroupId(),
          0,
          subVector.get());
    }
    auto blockingReason =
        queues_[0]->enqueue(subVector, tableView.num_rows(), &future);
    if (blockingReason != exec::BlockingReason::kNotBlocked) {
      blockingReasons_.push_back(blockingReason);
      futures_.push_back(std::move(future));
    }
  }
}

exec::BlockingReason CudfLocalPartition::isBlocked(ContinueFuture* future) {
  if (!futures_.empty()) {
    auto blockingReason = blockingReasons_.front();
    *future = folly::collectAll(futures_.begin(), futures_.end()).unit();
    futures_.clear();
    blockingReasons_.clear();
    return blockingReason;
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfLocalPartition::noMoreInput() {
  Operator::noMoreInput();
  for (const auto& queue : queues_) {
    queue->noMoreData();
  }
}

bool CudfLocalPartition::isFinished() {
  if (!futures_.empty() || !noMoreInput_) {
    return false;
  }
  flushVectorPool();

  return true;
}

} // namespace facebook::velox::cudf_velox
