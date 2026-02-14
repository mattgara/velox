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
#include "velox/experimental/cudf/CudfQueryConfig.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

namespace facebook::velox::cudf_velox {
namespace {

bool getDebugFlag(char const* name) {
  auto* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void topNDebugLog(
    char const* nodeId,
    char const* stage,
    rmm::cuda_stream_view stream,
    int64_t rows,
    int64_t aux) {
  static const bool kDebugEnabled = getDebugFlag("VELOX_CUDF_TOPN_DEBUG");
  if (!kDebugEnabled) {
    return;
  }
  std::fprintf(
      stderr,
      "[CudfTopNDebug] node=%s stage=%s stream=%p rows=%lld aux=%lld\n",
      nodeId,
      stage,
      reinterpret_cast<const void*>(stream.value()),
      static_cast<long long>(rows),
      static_cast<long long>(aux));
  static const bool kDebugCudaCheckEnabled =
      getDebugFlag("VELOX_CUDF_DEBUG_CHECK_CUDA");
  if (kDebugCudaCheckEnabled) {
    auto const peek = cudaPeekAtLastError();
    std::fprintf(
        stderr,
        "[CudfTopNDebug] node=%s stage=%s peek=%d(%s)\n",
        nodeId,
        stage,
        static_cast<int>(peek),
        cudaGetErrorString(peek));
  }
  static const bool kDebugSyncEnabled =
      getDebugFlag("VELOX_CUDF_TOPN_DEBUG_SYNC");
  if (kDebugSyncEnabled) {
    auto const syncStatus = cudaStreamSynchronize(stream.value());
    std::fprintf(
        stderr,
        "[CudfTopNDebug] node=%s stage=%s syncStatus=%d(%s)\n",
        nodeId,
        stage,
        static_cast<int>(syncStatus),
        cudaGetErrorString(syncStatus));
  }
}
} // namespace

CudfTopN::CudfTopN(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNNode>& topNNode)
    : exec::Operator(
          driverCtx,
          topNNode->outputType(),
          operatorId,
          topNNode->id(),
          "CudfTopN"),
      NvtxHelper(
          nvtx3::rgb{175, 238, 238}, // Pale Turquoise
          operatorId,
          fmt::format("[{}]", topNNode->id())),
      count_(topNNode->count()),
      topNNode_(topNNode) {
  kBatchSize_ = driverCtx->queryConfig().get<int32_t>(
      CudfQueryConfig::kCudfTopNBatchSize, kBatchSize_);
  const auto numColumns{outputType_->children().size()};
  const auto numSortingKeys{topNNode->sortingKeys().size()};
  std::vector<bool> isSortingKey(numColumns);
  sortKeys_.reserve(numSortingKeys);
  columnOrder_.reserve(numSortingKeys);
  nullOrder_.reserve(numSortingKeys);

  for (int i = 0; i < numSortingKeys; ++i) {
    const auto channel =
        exec::exprToChannel(topNNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopN doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    isSortingKey[channel] = true;
    auto const& sortingOrder = topNNode->sortingOrders()[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

CudfVectorPtr CudfTopN::mergeTopK(
    std::vector<CudfVectorPtr> topNBatches,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  topNDebugLog(topNNode_->id().c_str(), "mergeTopK.begin", stream, k, 0);
  std::vector<cudf::table_view> tableViews;
  std::vector<rmm::cuda_stream_view> inputStreams;
  tableViews.reserve(topNBatches.size());
  inputStreams.reserve(topNBatches.size());
  for (const auto& batch : topNBatches) {
    if (!batch) {
      continue;
    }
    tableViews.push_back(batch->getTableView());
    inputStreams.push_back(batch->stream());
  }
  topNDebugLog(
      topNNode_->id().c_str(),
      "mergeTopK.beforeJoinStreams",
      stream,
      tableViews.size(),
      inputStreams.size());
  // Ensure all upstream batch-producing streams are visible on the merge stream.
  cudf::detail::join_streams(inputStreams, stream);
  topNDebugLog(
      topNNode_->id().c_str(),
      "mergeTopK.afterJoinStreams",
      stream,
      tableViews.size(),
      inputStreams.size());
  auto mergedTable =
      cudf::merge(tableViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);
  topNDebugLog(
      topNNode_->id().c_str(),
      "mergeTopK.afterMerge",
      stream,
      mergedTable ? mergedTable->num_rows() : 0,
      k);
  // slice it
  auto topk =
      cudf::split(
          mergedTable->view(), {std::min(k, mergedTable->num_rows())}, stream)
          .front();
  auto const size = topk.num_rows();
  topNDebugLog(topNNode_->id().c_str(), "mergeTopK.afterSplit", stream, size, k);
  return std::make_shared<CudfVector>(
      topNBatches[0]->pool(),
      outputType_,
      size,
      std::make_unique<cudf::table>(topk),
      stream);
}

std::unique_ptr<cudf::table> CudfTopN::getTopK(
    cudf::table_view const& values,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  topNDebugLog(topNNode_->id().c_str(), "getTopK.begin", stream, values.num_rows(), k);
  auto keys = values.select(sortKeys_);
  auto const indices =
      cudf::stable_sorted_order(keys, columnOrder_, nullOrder_, stream, mr);
  topNDebugLog(
      topNNode_->id().c_str(),
      "getTopK.afterStableSortedOrder",
      stream,
      indices ? indices->size() : 0,
      k);
  auto const kIndices =
      cudf::split(indices->view(), {std::min(k, indices->size())}, stream)
          .front();
  auto gathered = cudf::detail::gather(
      values,
      kIndices,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::detail::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  topNDebugLog(
      topNNode_->id().c_str(),
      "getTopK.afterGather",
      stream,
      gathered ? gathered->num_rows() : 0,
      k);
  return gathered;
}

// helper to get topk of a table
CudfVectorPtr CudfTopN::getTopKBatch(CudfVectorPtr cudfInput, int32_t k) {
  if (k == 0 || cudfInput->size() == 0) {
    return nullptr;
  }
  auto stream = cudfInput->stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto values = cudfInput->getTableView();
  auto result = getTopK(values, k, stream, mr);
  auto const size = result->num_rows();
  return std::make_shared<CudfVector>(
      cudfInput->pool(), cudfInput->type(), size, std::move(result), stream);
}

void CudfTopN::addInput(RowVectorPtr input) {
  if (count_ == 0 || input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  topNDebugLog(
      topNNode_->id().c_str(),
      "addInput.begin",
      cudfInput->stream(),
      input->size(),
      topNBatches_.size());
  // Take topk of each input, add to batch.
  // If got kBatchSize_ batches, concat batches and topk once.
  // During getOutput, concat batches and topk once.
  topNBatches_.push_back(getTopKBatch(cudfInput, count_));
  // sum of sizes of topNBatches_ >= count_, then concat and topk once.
  auto totalSize = std::accumulate(
      topNBatches_.begin(),
      topNBatches_.end(),
      0,
      [](int32_t sum, const auto& batch) {
        return sum + (batch ? batch->size() : 0);
      });
  topNDebugLog(
      topNNode_->id().c_str(),
      "addInput.afterGetTopKBatch",
      cudfInput->stream(),
      totalSize,
      topNBatches_.size());
  if (topNBatches_.size() >= kBatchSize_ and totalSize >= count_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = cudf::get_current_device_resource_ref();

    topNDebugLog(
        topNNode_->id().c_str(),
        "addInput.beforeMergeTopK",
        stream,
        totalSize,
        topNBatches_.size());
    auto result = mergeTopK(topNBatches_, count_, stream, mr);
    topNBatches_.clear();
    topNBatches_.push_back(std::move(result));
    topNDebugLog(
        topNNode_->id().c_str(),
        "addInput.afterMergeTopK",
        stream,
        topNBatches_.front() ? topNBatches_.front()->size() : 0,
        topNBatches_.size());
  }
}

RowVectorPtr CudfTopN::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (topNBatches_.empty()) {
    finished_ = noMoreInput_;
    return nullptr;
  }

  auto stream = topNBatches_[0]->stream();
  auto mr = cudf::get_current_device_resource_ref();
  topNDebugLog(
      topNNode_->id().c_str(),
      "getOutput.beforeMergeTopK",
      stream,
      topNBatches_.size(),
      count_);
  auto result = mergeTopK(topNBatches_, count_, stream, mr);
  topNBatches_.clear();
  finished_ = noMoreInput_ && topNBatches_.empty();
  topNDebugLog(
      topNNode_->id().c_str(),
      "getOutput.afterMergeTopK",
      stream,
      result ? result->size() : 0,
      finished_ ? 1 : 0);
  return result;
}

void CudfTopN::noMoreInput() {
  Operator::noMoreInput();
  if (topNBatches_.empty()) {
    finished_ = true;
    return;
  }
}

bool CudfTopN::isFinished() {
  return finished_;
}
} // namespace facebook::velox::cudf_velox
