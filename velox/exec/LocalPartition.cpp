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

#include "velox/exec/LocalPartition.h"
#include "velox/common/Casts.h"
#include "velox/exec/Task.h"
#include "velox/vector/EncodedVectorCopy.h"

#ifdef PRESTO_ENABLE_CUDF
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/utilities/type_checks.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
#endif

namespace facebook::velox::exec {
namespace {
void notify(std::vector<ContinuePromise>& promises) {
  for (auto& promise : promises) {
    promise.setValue();
  }
}

#ifdef PRESTO_ENABLE_CUDF
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

struct ExchangeKeyStats {
  int64_t rows{0};
  int64_t stateRows{0};
  __int128_t sum{0};
  int64_t count{0};
  bool stateValid{false};
};

void trackLocalExchangeKeys(
    const RowVectorPtr& data,
    int partition,
    std::string const& taskId,
    std::string const& planNodeId,
    int32_t operatorId,
    uint32_t splitGroupId) {
  auto const& keys =
      facebook::velox::cudf_velox::CudfConfig::getInstance()
          .debugHashAggTrackKeys;
  if (keys.empty()) {
    return;
  }
  auto cudfVector =
      std::dynamic_pointer_cast<facebook::velox::cudf_velox::CudfVector>(data);
  if (!cudfVector) {
    return;
  }
  auto tableView = cudfVector->getTableView();
  if (tableView.num_columns() < 2) {
    return;
  }
  auto keyCol = tableView.column(0);
  auto keyType = keyCol.type().id();
  if (keyType != cudf::type_id::INT64 &&
      keyType != cudf::type_id::INT32) {
    return;
  }

  int32_t stateIndex = -1;
  for (cudf::size_type i = 1; i < tableView.num_columns(); ++i) {
    if (tableView.column(i).type().id() == cudf::type_id::STRING) {
      stateIndex = static_cast<int32_t>(i);
      break;
    }
  }
  if (stateIndex < 0) {
    return;
  }

  auto stateCol = tableView.column(stateIndex);
  auto decoded = facebook::velox::cudf_velox::deserializeDecimalSumStateWithCount(
      stateCol, 0, cudfVector->stream());
  auto sumView = decoded.sum->view();
  auto countView = decoded.count->view();
  auto sumType = sumView.type().id();
  if (sumType != cudf::type_id::DECIMAL64 &&
      sumType != cudf::type_id::DECIMAL128) {
    return;
  }

  std::unordered_map<int64_t, ExchangeKeyStats> stats;
  stats.reserve(keys.size());
  for (auto key : keys) {
    stats.emplace(key, ExchangeKeyStats{});
  }

  int64_t totalRows = tableView.num_rows();
  auto const& cfg = facebook::velox::cudf_velox::CudfConfig::getInstance();
  if (cfg.debugHashAggEndToEndMaxRows > 0 &&
      totalRows > cfg.debugHashAggEndToEndMaxRows) {
    totalRows = cfg.debugHashAggEndToEndMaxRows;
  }
  int64_t batchRows = cfg.debugHashAggEndToEndBatchRows;
  if (batchRows <= 0) {
    batchRows = totalRows;
  }

  auto keyElementSize = cudf::size_of(keyCol.type());
  auto sumElementSize = cudf::size_of(sumView.type());
  auto countElementSize = cudf::size_of(countView.type());
  auto const* keyBase = static_cast<const uint8_t*>(keyCol.head()) +
      keyCol.offset() * keyElementSize;
  auto const* sumBase = static_cast<const uint8_t*>(sumView.head()) +
      sumView.offset() * sumElementSize;
  auto const* countBase = static_cast<const uint8_t*>(countView.head()) +
      countView.offset() * countElementSize;
  auto keyBitOffset = keyCol.offset();
  auto sumBitOffset = sumView.offset();

  std::vector<uint8_t> keyMask;
  std::vector<uint8_t> sumMask;
  if (keyCol.null_count() > 0) {
    auto const maskBytes = static_cast<size_t>(
        cudf::bitmask_allocation_size_bytes(keyCol.offset() + keyCol.size()));
    keyMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        keyMask.data(),
        keyCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        cudfVector->stream().value());
    if (copyStatus != cudaSuccess) {
      return;
    }
  }
  if (sumView.null_count() > 0) {
    auto const maskBytes = static_cast<size_t>(
        cudf::bitmask_allocation_size_bytes(sumView.offset() + sumView.size()));
    sumMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        sumMask.data(),
        sumView.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        cudfVector->stream().value());
    if (copyStatus != cudaSuccess) {
      return;
    }
  }
  if (!keyMask.empty() || !sumMask.empty()) {
    auto const syncStatus = cudaStreamSynchronize(cudfVector->stream().value());
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
    auto const sumBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(sumElementSize);
    auto const countBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(countElementSize);
    keyHost.resize(keyBytes);
    sumHost.resize(sumBytes);
    countHost.resize(countBytes);

    auto const* keyPtr = keyBase + start * keyElementSize;
    auto const* sumPtr = sumBase + start * sumElementSize;
    auto const* countPtr = countBase + start * countElementSize;
    auto keyStatus = cudaMemcpyAsync(
        keyHost.data(),
        keyPtr,
        keyBytes,
        cudaMemcpyDeviceToHost,
        cudfVector->stream().value());
    auto sumStatus = cudaMemcpyAsync(
        sumHost.data(),
        sumPtr,
        sumBytes,
        cudaMemcpyDeviceToHost,
        cudfVector->stream().value());
    auto countStatus = cudaMemcpyAsync(
        countHost.data(),
        countPtr,
        countBytes,
        cudaMemcpyDeviceToHost,
        cudfVector->stream().value());
    if (keyStatus != cudaSuccess || sumStatus != cudaSuccess ||
        countStatus != cudaSuccess) {
      return;
    }
    auto syncStatus = cudaStreamSynchronize(cudfVector->stream().value());
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
      if (!isValidAt(sumMask, sumBitOffset + rowIndex)) {
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
    LOG(INFO) << "[HashAggExchangeTrack] stage=dequeue"
              << " task=" << taskId
              << " plan=" << planNodeId
              << " op=" << operatorId
              << " split=" << splitGroupId
              << " partition=" << partition
              << " ptr=" << data.get()
              << " key=" << key
              << " rows=" << entry.rows
              << " stateRows=" << entry.stateRows
              << " stateCount=" << entry.count
              << " stateSum=" << toString128(entry.sum)
              << " stateValid=" << entry.stateValid
              << " stateIndex=" << stateIndex;
  }
}
#endif
} // namespace

bool LocalExchangeMemoryManager::increaseMemoryUsage(
    ContinueFuture* future,
    int64_t added) {
  std::lock_guard<std::mutex> l(mutex_);
  bufferedBytes_ += added;

  if (bufferedBytes_ >= maxBufferSize_) {
    promises_.emplace_back("LocalExchangeMemoryManager::updateMemoryUsage");
    *future = promises_.back().getSemiFuture();
    return true;
  }

  return false;
}

std::vector<ContinuePromise> LocalExchangeMemoryManager::decreaseMemoryUsage(
    int64_t removed) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    bufferedBytes_ -= removed;

    if (bufferedBytes_ < maxBufferSize_) {
      promises = std::move(promises_);
    }
  }
  return promises;
}

void LocalExchangeVectorPool::push(const RowVectorPtr& vector, int64_t size) {
  pool_.withWLock([&](auto& pool) {
    if (totalSize_ + size <= capacity_) {
      pool.emplace(vector, size);
      totalSize_ += size;
    }
  });
}

RowVectorPtr LocalExchangeVectorPool::pop() {
  return pool_.withWLock([&](auto& pool) -> RowVectorPtr {
    while (!pool.empty()) {
      auto [vector, size] = std::move(pool.front());
      pool.pop();
      totalSize_ -= size;
      VELOX_CHECK_GE(totalSize_, 0);
      if (vector.use_count() == 1) {
        return vector;
      }
    }
    VELOX_CHECK_EQ(totalSize_, 0);
    return nullptr;
  });
}

void LocalExchangeQueue::addProducer() {
  queue_.withWLock([&](auto& /*queue*/) {
    VELOX_CHECK(!noMoreProducers_, "addProducer called after noMoreProducers");
    ++pendingProducers_;
  });
}

void LocalExchangeQueue::noMoreProducers() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK(!noMoreProducers_, "noMoreProducers can be called only once");
    noMoreProducers_ = true;
    if (pendingProducers_ == 0) {
      // No more data will be produced.
      consumerPromises = std::move(consumerPromises_);
    }
  });
  notify(consumerPromises);
}

void LocalExchangeQueue::drain() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK(!closed_, "Queue is closed");
    ++drainedProducers_;
    VELOX_CHECK_LE(drainedProducers_, pendingProducers_);
    if (drainedProducers_ != pendingProducers_) {
      return;
    }
    consumerPromises = std::move(consumerPromises_);
  });
  notify(consumerPromises);
}

BlockingReason LocalExchangeQueue::enqueue(
    RowVectorPtr input,
    int64_t inputBytes,
    ContinueFuture* future) {
  std::vector<ContinuePromise> consumerPromises;
  bool blockedOnConsumer = false;
  const bool isClosed = queue_.withWLock([&](auto& queue) {
    if (closed_) {
      return true;
    }
    queue.emplace(std::move(input), inputBytes);
    consumerPromises = std::move(consumerPromises_);

    if (memoryManager_->increaseMemoryUsage(future, inputBytes)) {
      blockedOnConsumer = true;
    }

    return false;
  });

  if (isClosed) {
    return BlockingReason::kNotBlocked;
  }

  notify(consumerPromises);

  if (blockedOnConsumer) {
    return BlockingReason::kWaitForConsumer;
  }

  return BlockingReason::kNotBlocked;
}

void LocalExchangeQueue::noMoreData() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK_EQ(drainedProducers_, 0);
    VELOX_CHECK_GT(pendingProducers_, 0);
    --pendingProducers_;
    if (noMoreProducers_ && pendingProducers_ == 0) {
      consumerPromises = std::move(consumerPromises_);
    }
  });
  notify(consumerPromises);
}

BlockingReason LocalExchangeQueue::next(
    ContinueFuture* future,
    memory::MemoryPool* pool,
    RowVectorPtr* data,
    bool& drained) {
  drained = false;
  int64_t size{0};
  std::vector<ContinuePromise> memoryPromises;
  const auto blockingReason = queue_.withWLock([&](auto& queue) {
    *data = nullptr;
    if (queue.empty()) {
      if (isFinishedLocked(queue)) {
        return BlockingReason::kNotBlocked;
      }
      if (testAndClearDrainedLocked()) {
        drained = true;
        return BlockingReason::kNotBlocked;
      }

      consumerPromises_.emplace_back("LocalExchangeQueue::next");
      *future = consumerPromises_.back().getSemiFuture();

      return BlockingReason::kWaitForProducer;
    }

    std::tie(*data, size) = std::move(queue.front());
    queue.pop();

    memoryPromises = memoryManager_->decreaseMemoryUsage(size);
    return BlockingReason::kNotBlocked;
  });

  notify(memoryPromises);
  if (*data != nullptr) {
    vectorPool_->push(*data, size);
  }
  return blockingReason;
}

bool LocalExchangeQueue::isFinishedLocked(const Queue& queue) const {
  if (closed_) {
    return true;
  }

  if (noMoreProducers_ && pendingProducers_ == 0 && queue.empty()) {
    return true;
  }

  return false;
}

bool LocalExchangeQueue::testAndClearDrainedLocked() {
  VELOX_CHECK(!closed_);
  VELOX_CHECK_GT(pendingProducers_, 0);
  if (pendingProducers_ != drainedProducers_) {
    return false;
  }
  drainedProducers_ = 0;
  return true;
}

bool LocalExchangeQueue::isFinished() {
  return queue_.withWLock([&](auto& queue) { return isFinishedLocked(queue); });
}

bool LocalExchangeQueue::testingProducersDone() const {
  return queue_.withRLock(
      [&](auto& queue) { return noMoreProducers_ && pendingProducers_ == 0; });
}

void LocalExchangeQueue::close() {
  std::vector<ContinuePromise> consumerPromises;
  std::vector<ContinuePromise> memoryPromises;
  queue_.withWLock([&](auto& queue) {
    uint64_t freedBytes = 0;
    while (!queue.empty()) {
      freedBytes += queue.front().second;
      queue.pop();
    }

    if (freedBytes) {
      memoryPromises = memoryManager_->decreaseMemoryUsage(freedBytes);
    }

    consumerPromises = std::move(consumerPromises_);
    closed_ = true;
  });
  notify(consumerPromises);
  notify(memoryPromises);
}

LocalExchange::LocalExchange(
    int32_t operatorId,
    DriverCtx* ctx,
    RowTypePtr outputType,
    const std::string& planNodeId,
    int partition)
    : SourceOperator(
          ctx,
          std::move(outputType),
          operatorId,
          planNodeId,
          "LocalExchange"),
      partition_{partition},
      queue_{operatorCtx_->task()->getLocalExchangeQueue(
          ctx->splitGroupId,
          planNodeId,
          partition)} {}

BlockingReason LocalExchange::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    auto reason = blockingReason_;
    blockingReason_ = BlockingReason::kNotBlocked;
    return reason;
  }

  return BlockingReason::kNotBlocked;
}

RowVectorPtr LocalExchange::getOutput() {
  if (hasDrained()) {
    return nullptr;
  }

  RowVectorPtr data;
  bool drained{false};
  blockingReason_ = queue_->next(&future_, pool(), &data, drained);
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    VELOX_CHECK(future_.valid());
    VELOX_CHECK(!drained);
    return nullptr;
  }

  if (data != nullptr) {
    VELOX_CHECK(!drained);
    auto lockedStats = stats_.wlock();
    lockedStats->addInputVector(data->estimateFlatSize(), data->size());
#ifdef PRESTO_ENABLE_CUDF
    trackLocalExchangeKeys(
        data, partition_, taskId(), planNodeId(), operatorId(), splitGroupId());
#endif
    return data;
  }

  if (drained) {
    VELOX_CHECK(!isDraining());
    operatorCtx_->driver()->drainOutput();
  } else {
    VELOX_CHECK(queue_->isFinished());
  }
  return nullptr;
}

bool LocalExchange::isFinished() {
  return queue_->isFinished();
}

void LocalExchange::close() {
  Operator::close();
  if (queue_) {
    queue_->close();
  }
}

LocalPartition::LocalPartition(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::LocalPartitionNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "LocalPartition"),
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()},
      partitionFunction_(
          numPartitions_ == 1 ? nullptr
                              : planNode->partitionFunctionSpec().create(
                                    numPartitions_,
                                    /*localExchange=*/true)),
      singlePartitionBufferSize_{
          (numPartitions_ <
               ctx->queryConfig()
                   .minLocalExchangePartitionCountToUsePartitionBuffer() ||
           eagerFlush)
              ? 0
              : ctx->queryConfig().maxLocalExchangePartitionBufferSize()},
      partitionBufferPreserveEncoding_{
          ctx->queryConfig().localExchangePartitionBufferPreserveEncoding()} {
  VELOX_CHECK(numPartitions_ == 1 || partitionFunction_ != nullptr);
  for (auto& queue : queues_) {
    queue->addProducer();
  }
  if (numPartitions_ > 0) {
    indexBuffers_.resize(numPartitions_);
    rawIndices_.resize(numPartitions_);
  }
}

void LocalPartition::allocateIndexBuffers(
    const std::vector<vector_size_t>& sizes) {
  VELOX_CHECK_EQ(indexBuffers_.size(), sizes.size());
  VELOX_CHECK_EQ(rawIndices_.size(), sizes.size());

  for (auto i = 0; i < sizes.size(); ++i) {
    const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
    if ((indexBuffers_[i] == nullptr) ||
        (indexBuffers_[i]->capacity() < indicesBufferBytes) ||
        !indexBuffers_[i]->unique()) {
      indexBuffers_[i] = allocateIndices(sizes[i], pool());
    } else {
      const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
      indexBuffers_[i]->setSize(indicesBufferBytes);
    }
    rawIndices_[i] = indexBuffers_[i]->asMutable<vector_size_t>();
  }
}

RowVectorPtr LocalPartition::wrapChildren(
    const RowVectorPtr& input,
    vector_size_t size,
    const BufferPtr& indices,
    RowVectorPtr reusable) {
  RowVectorPtr result;
  if (!reusable) {
    result = std::make_shared<RowVector>(
        pool(),
        input->type(),
        nullptr,
        size,
        std::vector<VectorPtr>(input->childrenSize()));
  } else {
    VELOX_CHECK(!reusable->mayHaveNulls());
    VELOX_CHECK_EQ(reusable.use_count(), 1);
    reusable->unsafeResize(size);
    result = std::move(reusable);
  }
  VELOX_CHECK_NOT_NULL(result);

  for (auto i = 0; i < input->childrenSize(); ++i) {
    auto& child = result->childAt(i);
    if (child && child->encoding() == VectorEncoding::Simple::DICTIONARY &&
        child.use_count() == 1) {
      child->BaseVector::resize(size);
      child->setWrapInfo(indices);
      child->setValueVector(input->childAt(i));
    } else {
      child = BaseVector::wrapInDictionary(
          nullptr, indices, size, input->childAt(i));
    }
  }

  result->updateContainsLazyNotLoaded();
  return result;
}

void LocalPartition::copy(
    const RowVectorPtr& input,
    const folly::Range<const BaseVector::CopyRange*>& ranges,
    const size_t partition,
    VectorPtr& target) {
  if (ranges.empty()) {
    return;
  }

  if (partitionBufferPreserveEncoding_) {
    encodedVectorCopy(
        EncodedVectorCopyOptions{pool(), false, 0.5}, input, ranges, target);
    return;
  }

  if (!target) {
    target = getOrCreateVector(partition);
  }
  target->resize(target->size() + ranges.size());
  target->copyRanges(input.get(), ranges);
}

VectorPtr LocalPartition::getOrCreateVector(const size_t partition) {
  auto reusable = queues_[partition]->getVector();
  if (reusable) {
    VELOX_CHECK_EQ(reusable->type(), outputType_);
    reusable->unsafeResize(0);
    for (auto i = 0; i < reusable->childrenSize(); ++i) {
      reusable->childAt(i) = nullptr;
    }
    return reusable;
  } else {
    return BaseVector::create<RowVector>(outputType_, 0, pool());
  }
}

void LocalPartition::populatePartitionBuffer(
    const RowVectorPtr& input,
    const vector_size_t numPartitionRows,
    const size_t partition,
    const vector_size_t* rawIndices,
    uint64_t& totalPartitionBufferSizeExcludingString,
    uint64_t& totalPartitionStringBufferSize) {
  VELOX_CHECK_GT(singlePartitionBufferSize_, 0);
  copyRanges_.resize(numPartitionRows);

  auto& partitionBuffer = partitionBuffers_[partition];
  auto targetIndex = 0;
  if (partitionBuffer) {
    targetIndex = partitionBuffer->size();
  }
  for (int i = 0; i < numPartitionRows; i++) {
    copyRanges_[i] = {rawIndices[i], targetIndex, 1};
    targetIndex++;
  }

  copy(input, copyRanges_, partition, partitionBuffer);

  if (partitionBuffer) {
    uint64_t stringBufferSize{0};
    auto totalSize = partitionBuffer->retainedSize(stringBufferSize);
    totalPartitionBufferSizeExcludingString += totalSize - stringBufferSize;
    totalPartitionStringBufferSize += stringBufferSize;
  }
}

RowVectorPtr LocalPartition::createPartition(
    const RowVectorPtr& input,
    const vector_size_t numPartitionRows,
    const size_t partition,
    const BufferPtr& indices) {
  RowVectorPtr partitionData{nullptr};
  if (singlePartitionBufferSize_ > 0) {
    auto& partitionBuffer = partitionBuffers_[partition];
    if (partitionBuffer) {
      partitionData =
          checkedPointerCast<RowVector, BaseVector>(partitionBuffer);
      partitionBuffers_[partition] = nullptr;
    }
  } else if (numPartitionRows > 0) {
    partitionData = wrapChildren(
        input, numPartitionRows, indices, queues_[partition]->getVector());
  }
  return partitionData;
}

void LocalPartition::populateAndEnqueuePartitions(
    RowVectorPtr input,
    const std::vector<vector_size_t>& numRowsPerPartition,
    const std::vector<BufferPtr>& indexBuffers,
    const std::vector<vector_size_t*>& rawIndicesBuffers) {
  uint64_t totalPartitionBufferSizeExcludingString = 0;
  uint64_t totalPartitionStringBufferSize = 0;
  uint16_t nonEmptyPartitionCount = 0;

  // Populate partition buffers if in buffer mode.
  if (singlePartitionBufferSize_ > 0) {
    if (partitionBuffers_.empty()) {
      partitionBuffers_.resize(numPartitions_);
    }
    for (auto partition = 0; partition < numPartitions_; partition++) {
      populatePartitionBuffer(
          input,
          numRowsPerPartition[partition],
          partition,
          rawIndicesBuffers[partition],
          totalPartitionBufferSizeExcludingString,
          totalPartitionStringBufferSize);
      if (partitionBuffers_[partition]) {
        nonEmptyPartitionCount++;
      }
    }
  } else {
    nonEmptyPartitionCount = numPartitions_ -
        std::count(numRowsPerPartition.begin(), numRowsPerPartition.end(), 0);
  }
  VELOX_CHECK_GT(
      nonEmptyPartitionCount,
      0,
      "Input rows should be assigned to at least one partition");

  // Calculate the partition buffer size across all partitions with amortized
  // string buffer sizes.
  auto balancedTotalPartitionBufferSize =
      totalPartitionBufferSizeExcludingString +
      (totalPartitionStringBufferSize / nonEmptyPartitionCount);
  auto inputRetainedSize = input->retainedSize();

  // Enqueue all partitions if one of the following conditions is met:
  // 1. This operator is not in buffer mode.
  // 2. This operator is in buffer mode and the total buffer size across all
  // partitions exceeds 'singlePartitionBufferSize_ * numPartitions_'.
  if (singlePartitionBufferSize_ == 0 ||
      balancedTotalPartitionBufferSize >=
          singlePartitionBufferSize_ * numPartitions_) {
    auto perPartitionAmortizedSize =
        (singlePartitionBufferSize_ > 0 ? balancedTotalPartitionBufferSize
                                        : inputRetainedSize) /
        nonEmptyPartitionCount;
    for (auto partition = 0; partition < numPartitions_; partition++) {
      auto partitionSize = numRowsPerPartition[partition];
      auto partitionData = createPartition(
          input, partitionSize, partition, indexBuffers[partition]);
      if (!partitionData) {
        continue;
      }

      ContinueFuture future;
      auto reason = queues_[partition]->enqueue(
          std::move(partitionData), perPartitionAmortizedSize, &future);
      if (reason != BlockingReason::kNotBlocked) {
        blockingReasons_.push_back(reason);
        futures_.push_back(std::move(future));
      }
    }
  }
}

void LocalPartition::addInput(RowVectorPtr input) {
  prepareForInput(input);
  if (input->size() == 0) {
    return;
  }

  const auto singlePartition = numPartitions_ == 1
      ? 0
      : partitionFunction_->partition(*input, partitions_);
  if (singlePartition.has_value()) {
    ContinueFuture future;
    auto blockingReason = queues_[singlePartition.value()]->enqueue(
        input, input->retainedSize(), &future);
    if (blockingReason != BlockingReason::kNotBlocked) {
      blockingReasons_.push_back(blockingReason);
      futures_.push_back(std::move(future));
    }
    return;
  }

  const auto numInput = input->size();
  std::vector<vector_size_t> maxIndex(numPartitions_, 0);
  for (auto i = 0; i < numInput; ++i) {
    ++maxIndex[partitions_[i]];
  }
  allocateIndexBuffers(maxIndex);

  std::fill(maxIndex.begin(), maxIndex.end(), 0);
  for (auto i = 0; i < numInput; ++i) {
    auto partition = partitions_[i];
    rawIndices_[partition][maxIndex[partition]] = i;
    ++maxIndex[partition];
  }

  populateAndEnqueuePartitions(input, maxIndex, indexBuffers_, rawIndices_);
}

void LocalPartition::prepareForInput(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  // Lazy vectors must be loaded or processed to ensure the late materialized in
  // order.
  for (auto& child : input->children()) {
    child->loadedVector();
  }
}

BlockingReason LocalPartition::isBlocked(ContinueFuture* future) {
  if (!futures_.empty()) {
    auto blockingReason = blockingReasons_.front();
    *future = folly::collectAll(futures_.begin(), futures_.end()).unit();
    futures_.clear();
    blockingReasons_.clear();
    return blockingReason;
  }
  return BlockingReason::kNotBlocked;
}

void LocalPartition::noMoreInput() {
  Operator::noMoreInput();
  if (!partitionBuffers_.empty()) {
    uint64_t totalPartitionBufferSizeExcludingString = 0;
    uint64_t totalPartitionStringBufferSize = 0;
    uint16_t nonEmptyPartitionCount = 0;
    for (auto partition = 0; partition < numPartitions_; partition++) {
      if (partitionBuffers_[partition]) {
        uint64_t stringBufferSize{0};
        auto totalSize =
            partitionBuffers_[partition]->retainedSize(stringBufferSize);
        totalPartitionBufferSizeExcludingString += totalSize - stringBufferSize;
        totalPartitionStringBufferSize += stringBufferSize;
        nonEmptyPartitionCount++;
      }
    }
    if (nonEmptyPartitionCount > 0) {
      auto balancedPartitionBufferSize =
          totalPartitionBufferSizeExcludingString +
          (totalPartitionStringBufferSize / nonEmptyPartitionCount);
      for (auto partition = 0; partition < numPartitions_; partition++) {
        if (partitionBuffers_[partition]) {
          auto partitionData = checkedPointerCast<RowVector, BaseVector>(
              partitionBuffers_[partition]);
          ContinueFuture future;

          queues_[partition]->enqueue(
              partitionData,
              balancedPartitionBufferSize / nonEmptyPartitionCount,
              &future);
        }
        partitionBuffers_[partition] = nullptr;
      }
    }
    partitionBuffers_.resize(0);
    copyRanges_.resize(0);
  }
  for (const auto& queue : queues_) {
    queue->noMoreData();
  }
}

bool LocalPartition::isFinished() {
  if (!futures_.empty() || !noMoreInput_) {
    return false;
  }

  return true;
}

RowVectorPtr LocalPartition::getOutput() {
  if (!isDraining()) {
    return nullptr;
  }
  for (auto& queue : queues_) {
    queue->drain();
  }
  finishDrain();
  return nullptr;
}
} // namespace facebook::velox::exec
