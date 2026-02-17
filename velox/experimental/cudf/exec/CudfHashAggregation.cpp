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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using namespace facebook::velox;

inline bool hashAggDebugEnabled() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugOperatorFlow;
}

inline bool hashAggDebugSyncEnabled() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugOperatorFlowSync;
}

inline int32_t hashAggDebugDeviceSyncPoint() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugOperatorFlowDeviceSyncPoint;
}

inline int32_t hashAggDebugFakeGroupbyMode() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggFakeGroupbyMode;
}

inline std::string const& hashAggDebugProbeDumpDir() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggProbeDumpDir;
}

inline int32_t hashAggDebugProbeDumpMaxRows() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggProbeDumpMaxRows;
}

inline std::string const& hashAggDebugDumpDir() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggDumpDir;
}

inline int32_t hashAggDebugDumpMaxRows() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggDumpMaxRows;
}

inline bool hashAggDebugStateRoundtripValidate() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggStateRoundtripValidate;
}

inline int32_t hashAggDebugDecimalCpuAggregateMode() {
  return facebook::velox::cudf_velox::CudfConfig::getInstance()
      .debugHashAggDecimalCpuAggregateMode;
}

constexpr int32_t kDeviceSyncGroupGroupByCore{1};
constexpr int32_t kDeviceSyncGroupDecimalRequest{2};
constexpr int32_t kDeviceSyncGroupDecimalCompute{3};
constexpr int32_t kDeviceSyncGroupGlobalAgg{4};
constexpr int32_t kDeviceSyncGroupOperatorBoundary{5};
constexpr int32_t kDeviceSyncGroupGroupByRequestIsolation{6};

const char* stepName(core::AggregationNode::Step step) {
  switch (step) {
    case core::AggregationNode::Step::kPartial:
      return "partial";
    case core::AggregationNode::Step::kFinal:
      return "final";
    case core::AggregationNode::Step::kIntermediate:
      return "intermediate";
    case core::AggregationNode::Step::kSingle:
      return "single";
  }
  return "unknown";
}

std::string cudfTypeName(cudf::type_id id) {
  return "type_id_" + std::to_string(static_cast<int>(id));
}

const char* aggregationKindName(cudf::aggregation::Kind kind) {
  using Kind = cudf::aggregation::Kind;
  switch (kind) {
    case Kind::SUM:
      return "SUM";
    case Kind::MIN:
      return "MIN";
    case Kind::MAX:
      return "MAX";
    case Kind::COUNT_VALID:
      return "COUNT_VALID";
    case Kind::COUNT_ALL:
      return "COUNT_ALL";
    case Kind::MEAN:
      return "MEAN";
    default:
      return "OTHER";
  }
}

const char* cudaErrorNameSafe(cudaError_t err) {
  auto const* name = cudaGetErrorName(err);
  return name != nullptr ? name : "<unknown-cuda-error-name>";
}

const char* cudaErrorStringSafe(cudaError_t err) {
  auto const* text = cudaGetErrorString(err);
  return text != nullptr ? text : "<unknown-cuda-error-string>";
}

const char* deviceSyncGroupName(int32_t groupId) {
  switch (groupId) {
    case kDeviceSyncGroupGroupByCore:
      return "groupby-core";
    case kDeviceSyncGroupDecimalRequest:
      return "decimal-request";
    case kDeviceSyncGroupDecimalCompute:
      return "decimal-compute";
    case kDeviceSyncGroupGlobalAgg:
      return "global-agg";
    case kDeviceSyncGroupOperatorBoundary:
      return "operator-boundary";
    case kDeviceSyncGroupGroupByRequestIsolation:
      return "groupby-request-isolation";
    default:
      return "unknown-group";
  }
}

bool isDeviceSyncGroupEnabled(int32_t selectedGroup, int32_t groupId) {
  if (selectedGroup == 0) {
    return false;
  }
  if (selectedGroup == -1) {
    return true;
  }
  return selectedGroup == groupId;
}

int32_t checkpointGroupFromStage(const char* stage) {
  std::string_view const stageView = stage != nullptr ? stage : "";
  auto const startsWith = [&](std::string_view prefix) {
    return stageView.rfind(prefix, 0) == 0;
  };
  if (startsWith("HashAgg.doGroupByAggregation.aggregateProbe")) {
    return kDeviceSyncGroupGroupByRequestIsolation;
  }
  if (startsWith("HashAgg.doGroupByAggregation")) {
    return kDeviceSyncGroupGroupByCore;
  }
  if (startsWith("Decimal.addGroupbyRequest")) {
    return kDeviceSyncGroupDecimalRequest;
  }
  if (startsWith("Decimal.doReduce") || startsWith("Decimal.computeAvgColumn")) {
    return kDeviceSyncGroupDecimalCompute;
  }
  if (startsWith("HashAgg.doGlobalAggregation")) {
    return kDeviceSyncGroupGlobalAgg;
  }
  return kDeviceSyncGroupOperatorBoundary;
}

std::mutex& knownStreamsMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<cudaStream_t>& knownStreams() {
  static std::vector<cudaStream_t> streams;
  return streams;
}

void registerKnownStream(cudaStream_t stream) {
  if (stream == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(knownStreamsMutex());
  auto& streams = knownStreams();
  if (std::find(streams.begin(), streams.end(), stream) == streams.end()) {
    streams.push_back(stream);
  }
}

void registerKnownStreams(rmm::cuda_stream_view stream) {
  registerKnownStream(stream.value());
  registerKnownStream(cudf::get_default_stream().value());
}

std::vector<cudaStream_t> getKnownStreamsSnapshot(cudaStream_t focusStream) {
  std::vector<cudaStream_t> snapshot;
  {
    std::lock_guard<std::mutex> lock(knownStreamsMutex());
    snapshot = knownStreams();
  }
  if (focusStream != nullptr &&
      std::find(snapshot.begin(), snapshot.end(), focusStream) ==
          snapshot.end()) {
    snapshot.push_back(focusStream);
  }
  return snapshot;
}

void logDeviceSyncModeOnce(int32_t selectedGroup) {
  static std::atomic<bool> logged{false};
  if (selectedGroup == 0 || logged.exchange(true)) {
    return;
  }
  LOG(INFO)
      << "[CudfHashAggDebug] deviceSyncMode selectedGroup=" << selectedGroup
      << " semantics=0:none,-1:all,1:groupby-core,2:decimal-request,"
      << "3:decimal-compute,4:global-agg,5:operator-boundary,"
      << "6:groupby-request-isolation";
}

void logKnownStreamsState(
    const char* stage,
    const char* phase,
    core::AggregationNode::Step step,
    cudaStream_t focusStream,
    int32_t groupId,
    int32_t selectedGroup) {
  auto const streams = getKnownStreamsSnapshot(focusStream);
  auto const defaultStream = cudf::get_default_stream().value();
  LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << ".knownStreams." << phase
            << " step=" << stepName(step)
            << " selectedGroup=" << selectedGroup << " group=" << groupId
            << "(" << deviceSyncGroupName(groupId) << ")"
            << " focusStream=" << reinterpret_cast<const void*>(focusStream)
            << " defaultStream="
            << reinterpret_cast<const void*>(defaultStream)
            << " knownStreamCount=" << streams.size();
  for (size_t idx = 0; idx < streams.size(); ++idx) {
    auto const stream = streams[idx];
    auto const queryErr = cudaStreamQuery(stream);
    LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << ".knownStreams."
              << phase << " step=" << stepName(step)
              << " selectedGroup=" << selectedGroup << " group=" << groupId
              << "(" << deviceSyncGroupName(groupId) << ")"
              << " streamIdx=" << idx
              << " stream=" << reinterpret_cast<const void*>(stream)
              << " isFocus=" << (stream == focusStream ? 1 : 0)
              << " isDefault=" << (stream == defaultStream ? 1 : 0)
              << " cudaStreamQuery=" << cudaErrorNameSafe(queryErr)
              << "(" << static_cast<int>(queryErr) << ")"
              << " cudaStreamQueryMsg=" << cudaErrorStringSafe(queryErr);
  }
}

void maybeDeviceSyncProbe(
    int32_t groupId,
    int32_t probeId,
    const char* stage,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    int64_t rows,
    int64_t cols,
    int64_t aux) {
  auto const selectedGroup = hashAggDebugDeviceSyncPoint();
  if (!isDeviceSyncGroupEnabled(selectedGroup, groupId)) {
    return;
  }
  logDeviceSyncModeOnce(selectedGroup);
  registerKnownStreams(stream);
  logKnownStreamsState(
      stage, "preDeviceSync", step, stream.value(), groupId, selectedGroup);
  auto const syncErr = cudaDeviceSynchronize();
  LOG(INFO) << "[CudfHashAggDebug] stage=" << stage
            << ".deviceSyncProbe group=" << groupId
            << "(" << deviceSyncGroupName(groupId) << ")"
            << " probeId=" << probeId << " step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " rows=" << rows << " cols=" << cols << " aux=" << aux
            << " selectedGroup=" << selectedGroup
            << " cudaDeviceSync=" << cudaErrorNameSafe(syncErr)
            << "(" << static_cast<int>(syncErr) << ")"
            << " cudaDeviceSyncMsg=" << cudaErrorStringSafe(syncErr);
  logKnownStreamsState(
      stage, "postDeviceSync", step, stream.value(), groupId, selectedGroup);
}

void logCudaCheckpoint(
    const char* stage,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    int64_t rows,
    int64_t cols,
    int64_t aux) {
  auto const selectedGroup = hashAggDebugDeviceSyncPoint();
  if (!hashAggDebugEnabled() && selectedGroup == 0) {
    return;
  }
  registerKnownStreams(stream);
  if (hashAggDebugEnabled()) {
    auto const peekErr = cudaPeekAtLastError();
    LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << " step="
              << stepName(step)
              << " stream=" << reinterpret_cast<const void*>(stream.value())
              << " rows=" << rows << " cols=" << cols << " aux=" << aux
              << " cudaPeek=" << cudaErrorNameSafe(peekErr)
              << "(" << static_cast<int>(peekErr) << ")"
              << " cudaPeekMsg=" << cudaErrorStringSafe(peekErr);
    if (hashAggDebugSyncEnabled()) {
      auto const syncErr = cudaStreamSynchronize(stream.value());
      LOG(INFO) << "[CudfHashAggDebug] stage=" << stage
                << ".streamSync step=" << stepName(step) << " stream="
                << reinterpret_cast<const void*>(stream.value()) << " rows="
                << rows << " cols=" << cols << " aux=" << aux
                << " cudaSync=" << cudaErrorNameSafe(syncErr)
                << "(" << static_cast<int>(syncErr) << ")"
                << " cudaSyncMsg=" << cudaErrorStringSafe(syncErr);
    }
  }
  maybeDeviceSyncProbe(
      checkpointGroupFromStage(stage),
      0,
      stage,
      step,
      stream,
      rows,
      cols,
      aux);
}

void logColumnViewDebug(
    const char* stage,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::column_view const& column,
    int64_t requestIdx,
    int depth = 0,
    int childIdx = -1) {
  if (!hashAggDebugEnabled()) {
    return;
  }
  auto const* headPtr = column.head<void>();
  auto const* nullMaskPtr = column.null_mask();
  auto const headAddr = reinterpret_cast<std::uintptr_t>(headPtr);
  auto const nullMaskAddr = reinterpret_cast<std::uintptr_t>(nullMaskPtr);
  int64_t elementSize = -1;
  std::uintptr_t dataAddr = 0;
  if (cudf::is_fixed_width(column.type())) {
    elementSize = static_cast<int64_t>(cudf::size_of(column.type()));
    if (headPtr != nullptr) {
      dataAddr = headAddr +
          static_cast<std::uintptr_t>(column.offset()) *
              static_cast<std::uintptr_t>(elementSize);
    }
  }
  LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << " step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " requestIdx=" << requestIdx << " depth=" << depth
            << " childIdx=" << childIdx << " size=" << column.size()
            << " nullCount=" << column.null_count() << " offset="
            << column.offset() << " numChildren=" << column.num_children()
            << " typeId=" << cudfTypeName(column.type().id())
            << " headPtr=" << headPtr
            << " dataPtr="
            << (dataAddr == 0 ? nullptr : reinterpret_cast<void const*>(dataAddr))
            << " nullMaskPtr=" << nullMaskPtr << " elementSize=" << elementSize
            << " headMod16="
            << (headPtr == nullptr ? -1 : static_cast<int64_t>(headAddr % 16))
            << " dataMod16="
            << (dataAddr == 0 ? -1 : static_cast<int64_t>(dataAddr % 16))
            << " nullMaskMod16="
            << (nullMaskPtr == nullptr
                    ? -1
                    : static_cast<int64_t>(nullMaskAddr % 16));
  constexpr int kMaxDepth = 5;
  if (depth >= kMaxDepth) {
    return;
  }
  for (int i = 0; i < column.num_children(); ++i) {
    logColumnViewDebug(
        stage, step, stream, column.child(i), requestIdx, depth + 1, i);
  }
}

void logGroupbyRequestsDebug(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& tableView,
    std::vector<cudf::groupby::aggregation_request> const& requests) {
  if (!hashAggDebugEnabled()) {
    return;
  }
  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.requests step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " inputRows=" << tableView.num_rows() << " inputCols="
            << tableView.num_columns() << " requestCount=" << requests.size();
  for (size_t i = 0; i < requests.size(); ++i) {
    auto const& request = requests[i];
    LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.requestMeta step="
              << stepName(step) << " stream="
              << reinterpret_cast<const void*>(stream.value()) << " requestIdx="
              << i << " aggregationCount=" << request.aggregations.size();
    logColumnViewDebug(
        "HashAgg.request.values", step, stream, request.values, i);
  }
}

void logGroupbyResultsDebug(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    std::vector<cudf::groupby::aggregation_result> const& results) {
  if (!hashAggDebugEnabled()) {
    return;
  }
  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.results step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " resultCount=" << results.size();
  for (size_t i = 0; i < results.size(); ++i) {
    auto const& result = results[i];
    LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.resultMeta step="
              << stepName(step) << " stream="
              << reinterpret_cast<const void*>(stream.value()) << " resultIdx="
              << i << " outputCount=" << result.results.size();
    for (size_t j = 0; j < result.results.size(); ++j) {
      auto const& col = result.results[j];
      if (!col) {
        LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.resultColumn.null step="
                  << stepName(step) << " stream="
                  << reinterpret_cast<const void*>(stream.value())
                  << " resultIdx=" << i << " outputIdx=" << j;
        continue;
      }
      logColumnViewDebug(
          "HashAgg.resultColumn.view", step, stream, col->view(), i, 0, j);
    }
  }
}

void logTableViewDebug(
    const char* stage,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& tableView) {
  if (!hashAggDebugEnabled()) {
    return;
  }
  LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << " step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " tableRows=" << tableView.num_rows() << " tableCols="
            << tableView.num_columns();
  for (int i = 0; i < tableView.num_columns(); ++i) {
    logColumnViewDebug(
        "HashAgg.tableView.column", step, stream, tableView.column(i), i);
  }
}

void logHashAggDebug(
    const char* stage,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    int64_t rows,
    int64_t cols,
    int64_t aux) {
  if (!hashAggDebugEnabled()) {
    return;
  }
  LOG(INFO) << "[CudfHashAggDebug] stage=" << stage << " step=" << stepName(step)
            << " stream=" << reinterpret_cast<const void*>(stream.value())
            << " rows=" << rows << " cols=" << cols << " aux=" << aux;
}

std::string formatRequestSubset(std::vector<size_t> const& subset) {
  std::string text{"["};
  for (size_t i = 0; i < subset.size(); ++i) {
    if (i > 0) {
      text += ",";
    }
    text += std::to_string(subset[i]);
  }
  text += "]";
  return text;
}

size_t bitCountU64(uint64_t value) {
  size_t count = 0;
  while (value != 0) {
    count += static_cast<size_t>(value & 1ULL);
    value >>= 1;
  }
  return count;
}

std::vector<std::vector<size_t>> buildRequestProbeSubsets(
    size_t requestCount,
    bool& exhaustive) {
  std::vector<std::vector<size_t>> subsets;
  if (requestCount == 0) {
    exhaustive = true;
    return subsets;
  }
  constexpr size_t kExhaustiveRequestCountLimit = 8;
  exhaustive = requestCount <= kExhaustiveRequestCountLimit;
  if (exhaustive) {
    auto const totalMasks = static_cast<uint64_t>(1ULL << requestCount);
    for (size_t subsetSize = 1; subsetSize <= requestCount; ++subsetSize) {
      for (uint64_t mask = 1; mask < totalMasks; ++mask) {
        if (bitCountU64(mask) != subsetSize) {
          continue;
        }
        std::vector<size_t> subset;
        subset.reserve(subsetSize);
        for (size_t bit = 0; bit < requestCount; ++bit) {
          if ((mask & (1ULL << bit)) != 0) {
            subset.push_back(bit);
          }
        }
        subsets.push_back(std::move(subset));
      }
    }
    return subsets;
  }

  // Large request counts can explode combinatorially; use a deterministic,
  // bounded subset list that still catches many interaction patterns.
  for (size_t i = 0; i < requestCount; ++i) {
    subsets.push_back({i});
  }
  constexpr size_t kPairProbeLimit = 32;
  size_t pairProbeCount = 0;
  for (size_t i = 0; i < requestCount && pairProbeCount < kPairProbeLimit; ++i) {
    for (size_t j = i + 1; j < requestCount && pairProbeCount < kPairProbeLimit;
         ++j) {
      subsets.push_back({i, j});
      ++pairProbeCount;
    }
  }
  for (size_t prefix = 3; prefix <= std::min<size_t>(requestCount, 8); ++prefix) {
    std::vector<size_t> subset;
    subset.reserve(prefix);
    for (size_t i = 0; i < prefix; ++i) {
      subset.push_back(i);
    }
    subsets.push_back(std::move(subset));
  }
  std::vector<size_t> allSubset;
  allSubset.reserve(requestCount);
  for (size_t i = 0; i < requestCount; ++i) {
    allSubset.push_back(i);
  }
  subsets.push_back(std::move(allSubset));
  return subsets;
}

cudf::groupby::aggregation_request cloneGroupbyRequest(
    cudf::groupby::aggregation_request const& source) {
  cudf::groupby::aggregation_request clone;
  clone.values = source.values;
  clone.aggregations.reserve(source.aggregations.size());
  for (auto const& aggregation : source.aggregations) {
    VELOX_CHECK_NOT_NULL(aggregation);
    auto aggregationClone = aggregation->clone();
    auto* groupbyAggregationClone =
        dynamic_cast<cudf::groupby_aggregation*>(aggregationClone.get());
    VELOX_CHECK_NOT_NULL(groupbyAggregationClone);
    aggregationClone.release();
    clone.aggregations.emplace_back(groupbyAggregationClone);
  }
  return clone;
}

std::vector<cudf::groupby::aggregation_request> cloneGroupbyRequestSubset(
    std::vector<cudf::groupby::aggregation_request> const& requests,
    std::vector<size_t> const& subset) {
  std::vector<cudf::groupby::aggregation_request> subsetRequests;
  subsetRequests.reserve(subset.size());
  for (auto const requestIdx : subset) {
    VELOX_CHECK_LT(requestIdx, requests.size());
    subsetRequests.emplace_back(cloneGroupbyRequest(requests[requestIdx]));
  }
  return subsetRequests;
}

struct CopiedProbeRequests {
  std::vector<std::unique_ptr<cudf::column>> copiedValues;
  std::vector<cudf::groupby::aggregation_request> requests;
};

CopiedProbeRequests buildCopiedGroupbyRequestSubset(
    std::vector<cudf::groupby::aggregation_request> const& requests,
    std::vector<size_t> const& subset,
    rmm::cuda_stream_view stream) {
  CopiedProbeRequests copied;
  copied.requests.reserve(subset.size());
  copied.copiedValues.reserve(subset.size());
  for (auto const requestIdx : subset) {
    VELOX_CHECK_LT(requestIdx, requests.size());
    auto const& sourceRequest = requests[requestIdx];
    auto copiedValues =
        std::make_unique<cudf::column>(sourceRequest.values, stream);
    cudf::groupby::aggregation_request copiedRequest;
    copiedRequest.values = copiedValues->view();
    copiedRequest.aggregations.reserve(sourceRequest.aggregations.size());
    for (auto const& aggregation : sourceRequest.aggregations) {
      VELOX_CHECK_NOT_NULL(aggregation);
      auto aggregationClone = aggregation->clone();
      auto* groupbyAggregationClone =
          dynamic_cast<cudf::groupby_aggregation*>(aggregationClone.get());
      VELOX_CHECK_NOT_NULL(groupbyAggregationClone);
      aggregationClone.release();
      copiedRequest.aggregations.emplace_back(groupbyAggregationClone);
    }
    copied.copiedValues.push_back(std::move(copiedValues));
    copied.requests.push_back(std::move(copiedRequest));
  }
  return copied;
}

struct AggregateProbeCallStatus {
  bool aggregateThrew{false};
  std::string aggregateException;
  int64_t outputRows{-1};
  int64_t outputCount{-1};
  cudaError_t peekErr{cudaSuccess};
  cudaError_t streamSyncErr{cudaSuccess};
  cudaError_t deviceSyncErr{cudaSuccess};
};

AggregateProbeCallStatus runAggregateProbeCall(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    cudf::null_policy nullPolicy,
    std::vector<cudf::groupby::aggregation_request>& probeRequests,
    size_t probeOrdinal,
    std::string const& subsetLabel,
    const char* mode) {
  AggregateProbeCallStatus status;
  try {
    cudf::groupby::groupby probeGroupBy(groupbyKeyView, nullPolicy);
    auto probeOutput = probeGroupBy.aggregate(probeRequests, stream);
    status.outputRows = probeOutput.first ? probeOutput.first->num_rows() : 0;
    status.outputCount = probeOutput.second.size();
  } catch (const std::exception& e) {
    status.aggregateThrew = true;
    status.aggregateException = e.what();
  }

  status.peekErr = cudaPeekAtLastError();
  status.streamSyncErr = cudaStreamSynchronize(stream.value());
  status.deviceSyncErr = cudaDeviceSynchronize();
  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "aggregateProbe.result step="
            << stepName(step) << " stream="
            << reinterpret_cast<const void*>(stream.value())
            << " probeOrdinal=" << probeOrdinal << " subset=" << subsetLabel
            << " mode=" << mode << " subsetSize=" << probeRequests.size()
            << " aggregateThrew=" << (status.aggregateThrew ? 1 : 0)
            << " outputRows=" << status.outputRows
            << " outputCount=" << status.outputCount
            << " cudaPeek=" << cudaErrorNameSafe(status.peekErr) << "("
            << static_cast<int>(status.peekErr) << ")"
            << " cudaPeekMsg=" << cudaErrorStringSafe(status.peekErr)
            << " cudaStreamSync=" << cudaErrorNameSafe(status.streamSyncErr)
            << "(" << static_cast<int>(status.streamSyncErr) << ")"
            << " cudaStreamSyncMsg=" << cudaErrorStringSafe(status.streamSyncErr)
            << " cudaDeviceSync=" << cudaErrorNameSafe(status.deviceSyncErr)
            << "(" << static_cast<int>(status.deviceSyncErr) << ")"
            << " cudaDeviceSyncMsg=" << cudaErrorStringSafe(status.deviceSyncErr)
            << (status.aggregateThrew ? " exception=" + status.aggregateException
                                      : "");
  return status;
}

bool aggregateProbeFailed(AggregateProbeCallStatus const& status) {
  return status.aggregateThrew || status.peekErr != cudaSuccess ||
      status.streamSyncErr != cudaSuccess || status.deviceSyncErr != cudaSuccess;
}

std::string formatAggregateProbeFailure(
    size_t probeOrdinal,
    std::string const& subsetLabel,
    const char* mode,
    AggregateProbeCallStatus const& status) {
  std::string failure = "probeOrdinal=" + std::to_string(probeOrdinal) +
      " subset=" + subsetLabel + " mode=" + mode + " aggregateThrew=" +
      (status.aggregateThrew ? "1" : "0") + " cudaPeek=" +
      std::string(cudaErrorNameSafe(status.peekErr)) + "(" +
      std::to_string(static_cast<int>(status.peekErr)) + ")" +
      " cudaStreamSync=" + std::string(cudaErrorNameSafe(status.streamSyncErr)) +
      "(" + std::to_string(static_cast<int>(status.streamSyncErr)) + ")" +
      " cudaDeviceSync=" + std::string(cudaErrorNameSafe(status.deviceSyncErr)) +
      "(" + std::to_string(static_cast<int>(status.deviceSyncErr)) + ")";
  if (status.aggregateThrew) {
    failure += " exception=" + status.aggregateException;
  }
  return failure;
}

struct FixedWidthColumnHostSnapshot {
  cudf::type_id typeId{cudf::type_id::EMPTY};
  int32_t scale{0};
  int64_t size{0};
  int64_t elementSize{0};
  int64_t nullCount{0};
  std::vector<uint8_t> data;
  std::vector<uint8_t> nullMask;
};

struct RequestHostSnapshot {
  size_t sourceRequestIdx{0};
  FixedWidthColumnHostSnapshot values;
  std::vector<cudf::aggregation::Kind> aggregationKinds;
};

struct AggregateProbeHostSnapshot {
  bool captured{false};
  std::string skipReason;
  std::vector<FixedWidthColumnHostSnapshot> keys;
  std::vector<RequestHostSnapshot> requests;
};

struct OutputTableHostSnapshot {
  bool captured{false};
  std::string skipReason;
  std::vector<FixedWidthColumnHostSnapshot> columns;
};

std::string sanitizeManifestValue(std::string text) {
  for (auto& ch : text) {
    if (ch == '\n' || ch == '\r') {
      ch = ' ';
    }
  }
  return text;
}

bool writeBinaryFile(
    std::filesystem::path const& path,
    std::vector<uint8_t> const& data,
    std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    error = "failed to open " + path.string();
    return false;
  }
  if (!data.empty()) {
    out.write(reinterpret_cast<char const*>(data.data()), data.size());
    if (!out.good()) {
      error = "failed to write " + path.string();
      return false;
    }
  }
  return true;
}

std::string captureFixedWidthColumnHost(
    cudf::column_view const& column,
    FixedWidthColumnHostSnapshot& snapshot) {
  if (!cudf::is_fixed_width(column.type())) {
    return "unsupported non-fixed-width type " +
        std::to_string(static_cast<int>(column.type().id()));
  }
  if (column.offset() != 0) {
    return "unsupported non-zero column offset " +
        std::to_string(column.offset());
  }
  auto const elementSize = cudf::size_of(column.type());
  if (elementSize <= 0) {
    return "invalid element size " + std::to_string(elementSize);
  }
  snapshot.typeId = column.type().id();
  snapshot.scale = column.type().scale();
  snapshot.size = column.size();
  snapshot.elementSize = elementSize;
  snapshot.nullCount = column.null_count();

  auto const dataBytes =
      static_cast<size_t>(snapshot.size) * static_cast<size_t>(elementSize);
  snapshot.data.assign(dataBytes, 0);
  if (dataBytes > 0) {
    auto const* dataPtr = column.data<uint8_t>();
    if (dataPtr == nullptr) {
      return "null fixed-width data pointer";
    }
    auto const copyStatus = cudaMemcpy(
        snapshot.data.data(),
        dataPtr,
        dataBytes,
        cudaMemcpyDeviceToHost);
    if (copyStatus != cudaSuccess) {
      return "cudaMemcpy data failed: " +
          std::string(cudaErrorNameSafe(copyStatus)) + "(" +
          std::to_string(static_cast<int>(copyStatus)) + ")";
    }
  }

  if (column.nullable()) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(column.size()));
    snapshot.nullMask.assign(maskBytes, 0);
    if (maskBytes > 0) {
      auto const* maskPtr = column.null_mask();
      if (maskPtr == nullptr) {
        return "nullable column has null null-mask pointer";
      }
      auto const copyStatus = cudaMemcpy(
          snapshot.nullMask.data(),
          maskPtr,
          maskBytes,
          cudaMemcpyDeviceToHost);
      if (copyStatus != cudaSuccess) {
        return "cudaMemcpy null-mask failed: " +
            std::string(cudaErrorNameSafe(copyStatus)) + "(" +
            std::to_string(static_cast<int>(copyStatus)) + ")";
      }
    }
  }
  return "";
}

std::string captureFixedWidthColumnHostAsync(
    rmm::cuda_stream_view stream,
    cudf::column_view const& column,
    FixedWidthColumnHostSnapshot& snapshot) {
  if (!cudf::is_fixed_width(column.type())) {
    return "unsupported non-fixed-width type " +
        std::to_string(static_cast<int>(column.type().id()));
  }
  if (column.offset() != 0) {
    return "unsupported non-zero column offset " +
        std::to_string(column.offset());
  }
  auto const elementSize = cudf::size_of(column.type());
  if (elementSize <= 0) {
    return "invalid element size " + std::to_string(elementSize);
  }
  snapshot.typeId = column.type().id();
  snapshot.scale = column.type().scale();
  snapshot.size = column.size();
  snapshot.elementSize = elementSize;
  snapshot.nullCount = column.null_count();

  auto const dataBytes =
      static_cast<size_t>(snapshot.size) * static_cast<size_t>(elementSize);
  snapshot.data.assign(dataBytes, 0);
  if (dataBytes > 0) {
    auto const* dataPtr = column.data<uint8_t>();
    if (dataPtr == nullptr) {
      return "null fixed-width data pointer";
    }
    auto const copyStatus = cudaMemcpyAsync(
        snapshot.data.data(),
        dataPtr,
        dataBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      return "cudaMemcpyAsync data failed: " +
          std::string(cudaErrorNameSafe(copyStatus)) + "(" +
          std::to_string(static_cast<int>(copyStatus)) + ")";
    }
  }

  if (column.nullable()) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(column.size()));
    snapshot.nullMask.assign(maskBytes, 0);
    if (maskBytes > 0) {
      auto const* maskPtr = column.null_mask();
      if (maskPtr == nullptr) {
        return "nullable column has null null-mask pointer";
      }
      auto const copyStatus = cudaMemcpyAsync(
          snapshot.nullMask.data(),
          maskPtr,
          maskBytes,
          cudaMemcpyDeviceToHost,
          stream.value());
      if (copyStatus != cudaSuccess) {
        return "cudaMemcpyAsync null-mask failed: " +
            std::string(cudaErrorNameSafe(copyStatus)) + "(" +
            std::to_string(static_cast<int>(copyStatus)) + ")";
      }
    }
  }
  return "";
}

AggregateProbeHostSnapshot captureAggregateProbeHostSnapshot(
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    std::vector<cudf::groupby::aggregation_request> const& probeRequests,
    std::vector<size_t> const& requestSubset,
    std::string const& dumpDir,
    int32_t dumpMaxRows) {
  AggregateProbeHostSnapshot snapshot;
  if (dumpDir.empty()) {
    snapshot.skipReason = "dump directory is empty";
    return snapshot;
  }
  if (dumpMaxRows > 0 && groupbyKeyView.num_rows() > dumpMaxRows) {
    snapshot.skipReason = "row count " +
        std::to_string(groupbyKeyView.num_rows()) +
        " exceeds dump max rows " + std::to_string(dumpMaxRows);
    return snapshot;
  }
  if (probeRequests.size() != requestSubset.size()) {
    snapshot.skipReason = "probe request count mismatch";
    return snapshot;
  }
  auto const syncStatus = cudaStreamSynchronize(stream.value());
  if (syncStatus != cudaSuccess) {
    snapshot.skipReason = "stream sync before host capture failed: " +
        std::string(cudaErrorNameSafe(syncStatus)) + "(" +
        std::to_string(static_cast<int>(syncStatus)) + ")";
    return snapshot;
  }

  snapshot.keys.reserve(groupbyKeyView.num_columns());
  for (int i = 0; i < groupbyKeyView.num_columns(); ++i) {
    FixedWidthColumnHostSnapshot keySnapshot;
    auto const captureError =
        captureFixedWidthColumnHost(groupbyKeyView.column(i), keySnapshot);
    if (!captureError.empty()) {
      snapshot.skipReason =
          "failed to capture grouping key column " + std::to_string(i) + ": " +
          captureError;
      snapshot.keys.clear();
      snapshot.requests.clear();
      return snapshot;
    }
    snapshot.keys.push_back(std::move(keySnapshot));
  }

  snapshot.requests.reserve(probeRequests.size());
  for (size_t i = 0; i < probeRequests.size(); ++i) {
    RequestHostSnapshot requestSnapshot;
    requestSnapshot.sourceRequestIdx = requestSubset[i];
    auto const captureError = captureFixedWidthColumnHost(
        probeRequests[i].values, requestSnapshot.values);
    if (!captureError.empty()) {
      snapshot.skipReason = "failed to capture request values " +
          std::to_string(i) + ": " + captureError;
      snapshot.keys.clear();
      snapshot.requests.clear();
      return snapshot;
    }
    requestSnapshot.aggregationKinds.reserve(probeRequests[i].aggregations.size());
    for (auto const& aggregation : probeRequests[i].aggregations) {
      if (!aggregation) {
        snapshot.skipReason =
            "request " + std::to_string(i) + " has null aggregation pointer";
        snapshot.keys.clear();
        snapshot.requests.clear();
        return snapshot;
      }
      requestSnapshot.aggregationKinds.push_back(aggregation->kind);
    }
    snapshot.requests.push_back(std::move(requestSnapshot));
  }

  snapshot.captured = true;
  return snapshot;
}

AggregateProbeHostSnapshot captureAggregateProbeHostSnapshotAsync(
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    std::vector<cudf::groupby::aggregation_request> const& probeRequests,
    std::vector<size_t> const& requestSubset,
    std::string const& dumpDir,
    int32_t dumpMaxRows) {
  AggregateProbeHostSnapshot snapshot;
  if (dumpDir.empty()) {
    snapshot.skipReason = "dump directory is empty";
    return snapshot;
  }
  if (dumpMaxRows > 0 && groupbyKeyView.num_rows() > dumpMaxRows) {
    snapshot.skipReason = "row count " +
        std::to_string(groupbyKeyView.num_rows()) +
        " exceeds dump max rows " + std::to_string(dumpMaxRows);
    return snapshot;
  }
  if (probeRequests.size() != requestSubset.size()) {
    snapshot.skipReason = "probe request count mismatch";
    return snapshot;
  }

  snapshot.keys.reserve(groupbyKeyView.num_columns());
  for (int i = 0; i < groupbyKeyView.num_columns(); ++i) {
    FixedWidthColumnHostSnapshot keySnapshot;
    auto const captureError =
        captureFixedWidthColumnHostAsync(stream, groupbyKeyView.column(i), keySnapshot);
    if (!captureError.empty()) {
      snapshot.skipReason =
          "failed to capture grouping key column " + std::to_string(i) + ": " +
          captureError;
      snapshot.keys.clear();
      snapshot.requests.clear();
      return snapshot;
    }
    snapshot.keys.push_back(std::move(keySnapshot));
  }

  snapshot.requests.reserve(probeRequests.size());
  for (size_t i = 0; i < probeRequests.size(); ++i) {
    RequestHostSnapshot requestSnapshot;
    requestSnapshot.sourceRequestIdx = requestSubset[i];
    auto const captureError = captureFixedWidthColumnHostAsync(
        stream, probeRequests[i].values, requestSnapshot.values);
    if (!captureError.empty()) {
      snapshot.skipReason = "failed to capture request values " +
          std::to_string(i) + ": " + captureError;
      snapshot.keys.clear();
      snapshot.requests.clear();
      return snapshot;
    }
    requestSnapshot.aggregationKinds.reserve(probeRequests[i].aggregations.size());
    for (auto const& aggregation : probeRequests[i].aggregations) {
      if (!aggregation) {
        snapshot.skipReason =
            "request " + std::to_string(i) + " has null aggregation pointer";
        snapshot.keys.clear();
        snapshot.requests.clear();
        return snapshot;
      }
      requestSnapshot.aggregationKinds.push_back(aggregation->kind);
    }
    snapshot.requests.push_back(std::move(requestSnapshot));
  }

  auto const syncStatus = cudaStreamSynchronize(stream.value());
  if (syncStatus != cudaSuccess) {
    snapshot.skipReason = "stream sync after host capture failed: " +
        std::string(cudaErrorNameSafe(syncStatus)) + "(" +
        std::to_string(static_cast<int>(syncStatus)) + ")";
    snapshot.keys.clear();
    snapshot.requests.clear();
    return snapshot;
  }

  snapshot.captured = true;
  return snapshot;
}

OutputTableHostSnapshot captureOutputTableHostSnapshotAsync(
    rmm::cuda_stream_view stream,
    cudf::table_view const& outputView) {
  OutputTableHostSnapshot snapshot;
  snapshot.columns.reserve(outputView.num_columns());
  for (auto const& column : outputView) {
    FixedWidthColumnHostSnapshot columnSnapshot;
    auto const captureError =
        captureFixedWidthColumnHostAsync(stream, column, columnSnapshot);
    if (!captureError.empty()) {
      snapshot.skipReason = captureError;
      snapshot.columns.clear();
      return snapshot;
    }
    snapshot.columns.push_back(std::move(columnSnapshot));
  }

  auto const syncStatus = cudaStreamSynchronize(stream.value());
  if (syncStatus != cudaSuccess) {
    snapshot.skipReason = "stream sync after host capture failed: " +
        std::string(cudaErrorNameSafe(syncStatus)) + "(" +
        std::to_string(static_cast<int>(syncStatus)) + ")";
    snapshot.columns.clear();
    return snapshot;
  }

  snapshot.captured = true;
  return snapshot;
}

std::string maybeWriteAggregateProbeDump(
    core::AggregationNode::Step step,
    size_t probeOrdinal,
    std::vector<size_t> const& requestSubset,
    const char* mode,
    cudf::null_policy nullPolicy,
    AggregateProbeCallStatus const& status,
    AggregateProbeHostSnapshot const& snapshot) {
  if (hashAggDebugProbeDumpDir().empty()) {
    return "";
  }
  if (!snapshot.captured) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "aggregateProbe.dumpSkipped step="
                 << stepName(step) << " probeOrdinal=" << probeOrdinal
                 << " subset=" << formatRequestSubset(requestSubset)
                 << " mode=" << mode
                 << " reason=" << sanitizeManifestValue(snapshot.skipReason);
    return "";
  }

  static std::atomic<uint64_t> dumpCounter{0};
  auto const tsMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  auto const dumpId = dumpCounter.fetch_add(1);
  auto const dumpDir =
      std::filesystem::path(hashAggDebugProbeDumpDir()) /
      ("hashagg_probe_" + std::to_string(tsMicros) + "_" +
       std::to_string(dumpId));

  std::error_code ec;
  std::filesystem::create_directories(dumpDir, ec);
  if (ec) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "aggregateProbe.dumpFailed step="
                 << stepName(step) << " probeOrdinal=" << probeOrdinal
                 << " subset=" << formatRequestSubset(requestSubset)
                 << " mode=" << mode
                 << " reason=create_directories failed: " << ec.message()
                 << " path=" << dumpDir.string();
    return "";
  }

  std::string fileError;
  for (size_t i = 0; i < snapshot.keys.size(); ++i) {
    auto const dataName = "key_" + std::to_string(i) + ".data.bin";
    if (!writeBinaryFile(dumpDir / dataName, snapshot.keys[i].data, fileError)) {
      LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                      "aggregateProbe.dumpFailed step="
                   << stepName(step) << " reason=" << fileError;
      return "";
    }
    if (!snapshot.keys[i].nullMask.empty()) {
      auto const maskName = "key_" + std::to_string(i) + ".nullmask.bin";
      if (!writeBinaryFile(
              dumpDir / maskName, snapshot.keys[i].nullMask, fileError)) {
        LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                        "aggregateProbe.dumpFailed step="
                     << stepName(step) << " reason=" << fileError;
        return "";
      }
    }
  }
  for (size_t i = 0; i < snapshot.requests.size(); ++i) {
    auto const dataName = "request_" + std::to_string(i) + ".data.bin";
    if (!writeBinaryFile(
            dumpDir / dataName, snapshot.requests[i].values.data, fileError)) {
      LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                      "aggregateProbe.dumpFailed step="
                   << stepName(step) << " reason=" << fileError;
      return "";
    }
    if (!snapshot.requests[i].values.nullMask.empty()) {
      auto const maskName = "request_" + std::to_string(i) + ".nullmask.bin";
      if (!writeBinaryFile(
              dumpDir / maskName,
              snapshot.requests[i].values.nullMask,
              fileError)) {
        LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                        "aggregateProbe.dumpFailed step="
                     << stepName(step) << " reason=" << fileError;
        return "";
      }
    }
  }

  auto const manifestPath = dumpDir / "manifest.txt";
  std::ofstream manifest(manifestPath);
  if (!manifest.is_open()) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "aggregateProbe.dumpFailed step="
                 << stepName(step) << " reason=failed to open manifest "
                 << manifestPath.string();
    return "";
  }

  auto const subsetLabel = formatRequestSubset(requestSubset);
  manifest << "format_version=1\n";
  manifest << "step=" << stepName(step) << "\n";
  manifest << "probe_ordinal=" << probeOrdinal << "\n";
  manifest << "mode=" << mode << "\n";
  manifest << "subset=" << subsetLabel << "\n";
  manifest << "null_policy=" << static_cast<int32_t>(nullPolicy) << "\n";
  manifest << "aggregate_threw=" << (status.aggregateThrew ? 1 : 0) << "\n";
  manifest << "aggregate_exception="
           << sanitizeManifestValue(status.aggregateException) << "\n";
  manifest << "cuda_peek=" << static_cast<int32_t>(status.peekErr) << "\n";
  manifest << "cuda_stream_sync=" << static_cast<int32_t>(status.streamSyncErr)
           << "\n";
  manifest << "cuda_device_sync=" << static_cast<int32_t>(status.deviceSyncErr)
           << "\n";
  manifest << "key_count=" << snapshot.keys.size() << "\n";
  manifest << "request_count=" << snapshot.requests.size() << "\n";
  for (size_t i = 0; i < snapshot.keys.size(); ++i) {
    auto const& key = snapshot.keys[i];
    manifest << "key." << i << ".type_id=" << static_cast<int32_t>(key.typeId)
             << "\n";
    manifest << "key." << i << ".scale=" << key.scale << "\n";
    manifest << "key." << i << ".size=" << key.size << "\n";
    manifest << "key." << i << ".element_size=" << key.elementSize << "\n";
    manifest << "key." << i << ".null_count=" << key.nullCount << "\n";
    manifest << "key." << i << ".data_file=key_" << i << ".data.bin\n";
    manifest << "key." << i << ".null_mask_file="
             << (key.nullMask.empty()
                     ? std::string()
                     : "key_" + std::to_string(i) + ".nullmask.bin")
             << "\n";
  }
  for (size_t i = 0; i < snapshot.requests.size(); ++i) {
    auto const& request = snapshot.requests[i];
    auto const& values = request.values;
    manifest << "request." << i << ".source_request_idx="
             << request.sourceRequestIdx << "\n";
    manifest << "request." << i << ".type_id="
             << static_cast<int32_t>(values.typeId) << "\n";
    manifest << "request." << i << ".scale=" << values.scale << "\n";
    manifest << "request." << i << ".size=" << values.size << "\n";
    manifest << "request." << i << ".element_size=" << values.elementSize
             << "\n";
    manifest << "request." << i << ".null_count=" << values.nullCount << "\n";
    manifest << "request." << i << ".data_file=request_" << i << ".data.bin\n";
    manifest << "request." << i << ".null_mask_file="
             << (values.nullMask.empty()
                     ? std::string()
                     : "request_" + std::to_string(i) + ".nullmask.bin")
             << "\n";
    manifest << "request." << i
             << ".aggregation_count=" << request.aggregationKinds.size() << "\n";
    for (size_t aggIdx = 0; aggIdx < request.aggregationKinds.size(); ++aggIdx) {
      auto const aggKind = request.aggregationKinds[aggIdx];
      manifest << "request." << i << ".aggregation_kind." << aggIdx << "="
               << static_cast<int32_t>(aggKind) << "\n";
      manifest << "request." << i << ".aggregation_kind_name." << aggIdx << "="
               << aggregationKindName(aggKind) << "\n";
    }
  }
  manifest << "replay_command=velox_cudf_hashagg_replay --manifest "
           << manifestPath.string() << "\n";
  manifest.close();
  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "aggregateProbe.dumpWritten step="
            << stepName(step) << " probeOrdinal=" << probeOrdinal
            << " subset=" << subsetLabel << " mode=" << mode
            << " dumpDir=" << dumpDir.string()
            << " replayCommand='velox_cudf_hashagg_replay --manifest "
            << manifestPath.string() << "'";
  return dumpDir.string();
}

std::string maybeWriteGroupbyDump(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    cudf::null_policy nullPolicy,
    std::vector<cudf::groupby::aggregation_request> const& requests,
    int64_t inputRows,
    int64_t inputCols,
    int64_t numGroupingKeys) {
  if (hashAggDebugDumpDir().empty()) {
    return "";
  }

  std::vector<size_t> requestSubset;
  requestSubset.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    requestSubset.push_back(i);
  }

  auto snapshot = captureAggregateProbeHostSnapshotAsync(
      stream,
      groupbyKeyView,
      requests,
      requestSubset,
      hashAggDebugDumpDir(),
      hashAggDebugDumpMaxRows());
  if (!snapshot.captured) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "groupbyDumpSkipped step="
                 << stepName(step) << " reason="
                 << sanitizeManifestValue(snapshot.skipReason);
    return "";
  }

  AggregateProbeCallStatus status;
  static std::atomic<uint64_t> dumpCounter{0};
  auto const baseDir = std::filesystem::path(hashAggDebugDumpDir());
  uint64_t dumpId = dumpCounter.fetch_add(1);
  std::filesystem::path dumpDir;
  for (int attempt = 0; attempt < 1024; ++attempt) {
    std::ostringstream name;
    name << "hashagg_main_" << std::setw(6) << std::setfill('0') << dumpId;
    dumpDir = baseDir / name.str();
    if (!std::filesystem::exists(dumpDir)) {
      break;
    }
    dumpId = dumpCounter.fetch_add(1);
  }

  std::error_code ec;
  std::filesystem::create_directories(dumpDir, ec);
  if (ec) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "groupbyDumpFailed step="
                 << stepName(step) << " reason=create_directories failed: "
                 << ec.message() << " path=" << dumpDir.string();
    return "";
  }

  std::string fileError;
  for (size_t i = 0; i < snapshot.keys.size(); ++i) {
    auto const dataName = "key_" + std::to_string(i) + ".data.bin";
    if (!writeBinaryFile(dumpDir / dataName, snapshot.keys[i].data, fileError)) {
      LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                      "groupbyDumpFailed step="
                   << stepName(step) << " reason=" << fileError;
      return "";
    }
    if (!snapshot.keys[i].nullMask.empty()) {
      auto const maskName = "key_" + std::to_string(i) + ".nullmask.bin";
      if (!writeBinaryFile(
              dumpDir / maskName, snapshot.keys[i].nullMask, fileError)) {
        LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                        "groupbyDumpFailed step="
                     << stepName(step) << " reason=" << fileError;
        return "";
      }
    }
  }
  for (size_t i = 0; i < snapshot.requests.size(); ++i) {
    auto const dataName = "request_" + std::to_string(i) + ".data.bin";
    if (!writeBinaryFile(
            dumpDir / dataName, snapshot.requests[i].values.data, fileError)) {
      LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                      "groupbyDumpFailed step="
                   << stepName(step) << " reason=" << fileError;
      return "";
    }
    if (!snapshot.requests[i].values.nullMask.empty()) {
      auto const maskName = "request_" + std::to_string(i) + ".nullmask.bin";
      if (!writeBinaryFile(
              dumpDir / maskName,
              snapshot.requests[i].values.nullMask,
              fileError)) {
        LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                        "groupbyDumpFailed step="
                     << stepName(step) << " reason=" << fileError;
        return "";
      }
    }
  }

  auto const manifestPath = dumpDir / "manifest.txt";
  std::ofstream manifest(manifestPath);
  if (!manifest.is_open()) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "groupbyDumpFailed step="
                 << stepName(step) << " reason=failed to open manifest "
                 << manifestPath.string();
    return "";
  }

  auto const subsetLabel = formatRequestSubset(requestSubset);
  manifest << "format_version=1\n";
  manifest << "dump_kind=main\n";
  manifest << "dump_sequence=" << dumpId << "\n";
  manifest << "dump_name=" << dumpDir.filename().string() << "\n";
  manifest << "step=" << stepName(step) << "\n";
  manifest << "probe_ordinal=0\n";
  manifest << "mode=main\n";
  manifest << "subset=" << subsetLabel << "\n";
  manifest << "null_policy=" << static_cast<int32_t>(nullPolicy) << "\n";
  manifest << "input_rows=" << inputRows << "\n";
  manifest << "input_cols=" << inputCols << "\n";
  manifest << "grouping_keys=" << numGroupingKeys << "\n";
  manifest << "aggregate_threw=" << (status.aggregateThrew ? 1 : 0) << "\n";
  manifest << "aggregate_exception="
           << sanitizeManifestValue(status.aggregateException) << "\n";
  manifest << "cuda_peek=" << static_cast<int32_t>(status.peekErr) << "\n";
  manifest << "cuda_stream_sync=" << static_cast<int32_t>(status.streamSyncErr)
           << "\n";
  manifest << "cuda_device_sync=" << static_cast<int32_t>(status.deviceSyncErr)
           << "\n";
  manifest << "key_count=" << snapshot.keys.size() << "\n";
  manifest << "request_count=" << snapshot.requests.size() << "\n";
  for (size_t i = 0; i < snapshot.keys.size(); ++i) {
    auto const& key = snapshot.keys[i];
    manifest << "key." << i << ".type_id=" << static_cast<int32_t>(key.typeId)
             << "\n";
    manifest << "key." << i << ".scale=" << key.scale << "\n";
    manifest << "key." << i << ".size=" << key.size << "\n";
    manifest << "key." << i << ".element_size=" << key.elementSize << "\n";
    manifest << "key." << i << ".null_count=" << key.nullCount << "\n";
    manifest << "key." << i << ".data_file=key_" << i << ".data.bin\n";
    manifest << "key." << i << ".null_mask_file="
             << (key.nullMask.empty()
                     ? std::string()
                     : "key_" + std::to_string(i) + ".nullmask.bin")
             << "\n";
  }
  for (size_t i = 0; i < snapshot.requests.size(); ++i) {
    auto const& values = snapshot.requests[i].values;
    manifest << "request." << i << ".source_request_idx="
             << snapshot.requests[i].sourceRequestIdx << "\n";
    manifest << "request." << i << ".type_id="
             << static_cast<int32_t>(values.typeId) << "\n";
    manifest << "request." << i << ".scale=" << values.scale << "\n";
    manifest << "request." << i << ".size=" << values.size << "\n";
    manifest << "request." << i << ".element_size=" << values.elementSize
             << "\n";
    manifest << "request." << i << ".null_count=" << values.nullCount << "\n";
    manifest << "request." << i << ".data_file=request_" << i << ".data.bin\n";
    manifest << "request." << i << ".null_mask_file="
             << (values.nullMask.empty()
                     ? std::string()
                     : "request_" + std::to_string(i) + ".nullmask.bin")
             << "\n";
    manifest << "request." << i << ".aggregation_count="
             << snapshot.requests[i].aggregationKinds.size() << "\n";
    for (size_t aggIdx = 0; aggIdx < snapshot.requests[i].aggregationKinds.size();
         ++aggIdx) {
      auto const kind = snapshot.requests[i].aggregationKinds[aggIdx];
      manifest << "request." << i << ".aggregation_kind." << aggIdx << "="
               << static_cast<int32_t>(kind) << "\n";
      manifest << "request." << i << ".aggregation_kind_name." << aggIdx << "="
               << aggregationKindName(kind) << "\n";
    }
  }
  manifest << "replay_command=velox_cudf_hashagg_replay --manifest "
           << manifestPath.string() << "\n";
  manifest.close();

  auto const indexPath = baseDir / "hashagg_dump_index.txt";
  std::ofstream indexFile(indexPath, std::ios::app);
  if (indexFile.is_open()) {
    indexFile << dumpDir.string() << "\n";
  }
  auto const latestPath = baseDir / "hashagg_dump_latest.txt";
  std::ofstream latestFile(latestPath);
  if (latestFile.is_open()) {
    latestFile << dumpDir.string() << "\n";
  }

  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "groupbyDumpWritten step="
            << stepName(step) << " dumpDir=" << dumpDir.string()
            << " replayCommand='velox_cudf_hashagg_replay --manifest "
            << manifestPath.string() << "'";
  return dumpDir.string();
}

std::string maybeWriteGroupbyOutputDump(
    std::string const& dumpDir,
    core::AggregationNode::Step step,
    int64_t numGroupingKeys,
    rmm::cuda_stream_view stream,
    cudf::table_view const& outputView) {
  if (dumpDir.empty()) {
    return "";
  }
  if (outputView.num_columns() == 0) {
    return "";
  }

  auto snapshot = captureOutputTableHostSnapshotAsync(stream, outputView);
  if (!snapshot.captured) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "outputDumpSkipped step="
                 << stepName(step) << " reason="
                 << sanitizeManifestValue(snapshot.skipReason);
    return "";
  }

  std::filesystem::path baseDir(dumpDir);
  std::string fileError;
  for (size_t i = 0; i < snapshot.columns.size(); ++i) {
    auto const dataName = "output_" + std::to_string(i) + ".data.bin";
    if (!writeBinaryFile(baseDir / dataName, snapshot.columns[i].data, fileError)) {
      LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                      "outputDumpFailed step="
                   << stepName(step) << " reason=" << fileError;
      return "";
    }
    if (!snapshot.columns[i].nullMask.empty()) {
      auto const maskName = "output_" + std::to_string(i) + ".nullmask.bin";
      if (!writeBinaryFile(
              baseDir / maskName, snapshot.columns[i].nullMask, fileError)) {
        LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                        "outputDumpFailed step="
                     << stepName(step) << " reason=" << fileError;
        return "";
      }
    }
  }

  auto const manifestPath = baseDir / "output_manifest.txt";
  std::ofstream manifest(manifestPath);
  if (!manifest.is_open()) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "outputDumpFailed step="
                 << stepName(step) << " reason=failed to open manifest "
                 << manifestPath.string();
    return "";
  }

  manifest << "format_version=1\n";
  manifest << "step=" << stepName(step) << "\n";
  manifest << "output_row_count=" << outputView.num_rows() << "\n";
  manifest << "output_column_count=" << snapshot.columns.size() << "\n";
  manifest << "output_key_count=" << numGroupingKeys << "\n";
  for (size_t i = 0; i < snapshot.columns.size(); ++i) {
    auto const& col = snapshot.columns[i];
    manifest << "output." << i << ".type_id="
             << static_cast<int32_t>(col.typeId) << "\n";
    manifest << "output." << i << ".scale=" << col.scale << "\n";
    manifest << "output." << i << ".size=" << col.size << "\n";
    manifest << "output." << i << ".element_size=" << col.elementSize << "\n";
    manifest << "output." << i << ".null_count=" << col.nullCount << "\n";
    manifest << "output." << i << ".data_file=output_" << i << ".data.bin\n";
    manifest << "output." << i << ".null_mask_file="
             << (col.nullMask.empty()
                     ? std::string()
                     : "output_" + std::to_string(i) + ".nullmask.bin")
             << "\n";
  }
  manifest.close();

  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "outputDumpWritten step="
            << stepName(step) << " dumpDir=" << dumpDir;
  return dumpDir;
}

struct EndToEndValidationResult {
  bool skipped{false};
  std::string reason;
  int64_t mismatches{0};
  int64_t missing{0};
  int64_t checked{0};
  int64_t expectedKeys{0};
  int64_t outputKeys{0};
  int64_t firstKey{0};
  __int128_t firstExpected{0};
  __int128_t firstActual{0};
};

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

std::string trimCopy(std::string const& input) {
  size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  size_t end = input.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(start, end - start);
}

bool parseInt64(std::string const& text, int64_t& out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  auto value = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return false;
  }
  out = static_cast<int64_t>(value);
  return true;
}

bool parseScaledDecimalToInt128(
    std::string const& text,
    int32_t scale,
    __int128_t& out) {
  if (text.empty()) {
    return false;
  }
  if (scale > 0) {
    return false;
  }
  std::string s = trimCopy(text);
  if (s.empty()) {
    return false;
  }
  bool negative = false;
  if (s[0] == '-') {
    negative = true;
    s.erase(0, 1);
  } else if (s[0] == '+') {
    s.erase(0, 1);
  }
  auto dotPos = s.find('.');
  std::string intPart = dotPos == std::string::npos ? s : s.substr(0, dotPos);
  std::string fracPart = dotPos == std::string::npos ? "" : s.substr(dotPos + 1);
  if (intPart.empty()) {
    intPart = "0";
  }
  for (char c : intPart) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  for (char c : fracPart) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  int32_t targetFrac = -scale;
  if (static_cast<int32_t>(fracPart.size()) > targetFrac) {
    return false;
  }
  int32_t padZeros = targetFrac - static_cast<int32_t>(fracPart.size());
  __int128_t value = 0;
  for (char c : intPart) {
    value = value * 10 + (c - '0');
  }
  for (char c : fracPart) {
    value = value * 10 + (c - '0');
  }
  for (int32_t i = 0; i < padZeros; ++i) {
    value *= 10;
  }
  out = negative ? -value : value;
  return true;
}

struct ExpectedEntry {
  __int128_t value{0};
};

struct ExpectedCache {
  std::string path;
  int32_t scale{0};
  std::unordered_map<int64_t, ExpectedEntry> values;
};

ExpectedCache loadExpectedValues(
    std::string const& path,
    int32_t scale) {
  ExpectedCache cache;
  cache.path = path;
  cache.scale = scale;
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open expected file: " + path);
  }
  std::string line;
  bool headerSkipped = false;
  while (std::getline(input, line)) {
    line = trimCopy(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::string keyText;
    std::string valueText;
    auto commaPos = line.find(',');
    if (commaPos != std::string::npos) {
      keyText = trimCopy(line.substr(0, commaPos));
      valueText = trimCopy(line.substr(commaPos + 1));
    } else {
      std::istringstream iss(line);
      iss >> keyText >> valueText;
    }
    int64_t key = 0;
    if (!parseInt64(keyText, key)) {
      if (!headerSkipped) {
        headerSkipped = true;
        continue;
      }
      throw std::runtime_error("failed to parse key: " + keyText);
    }
    __int128_t value = 0;
    if (!parseScaledDecimalToInt128(valueText, scale, value)) {
      throw std::runtime_error("failed to parse value: " + valueText);
    }
    cache.values.emplace(key, ExpectedEntry{value});
  }
  return cache;
}

EndToEndValidationResult validateOutputAgainstExpected(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& outputView,
    int64_t batchRows,
    std::string const& expectedPath) {
  EndToEndValidationResult result;
  if (step != core::AggregationNode::Step::kFinal &&
      step != core::AggregationNode::Step::kSingle) {
    result.skipped = true;
    result.reason = "expected validation only final/single";
    return result;
  }
  if (outputView.num_columns() != 2) {
    result.skipped = true;
    result.reason = "output column count != keys+1";
    return result;
  }
  auto outKeyCol = outputView.column(0);
  auto outValCol = outputView.column(1);
  if (!cudf::is_fixed_width(outKeyCol.type()) ||
      !cudf::is_fixed_width(outValCol.type())) {
    result.skipped = true;
    result.reason = "non-fixed-width output";
    return result;
  }
  auto outKeyType = outKeyCol.type().id();
  auto outValType = outValCol.type().id();
  if (outKeyType != cudf::type_id::INT64 &&
      outKeyType != cudf::type_id::INT32) {
    result.skipped = true;
    result.reason = "unsupported key type";
    return result;
  }
  if (outValType != cudf::type_id::DECIMAL64 &&
      outValType != cudf::type_id::DECIMAL128) {
    result.skipped = true;
    result.reason = "unsupported value type";
    return result;
  }

  static std::mutex expectedMutex;
  static std::shared_ptr<ExpectedCache> cachedExpected;
  std::shared_ptr<ExpectedCache> expected;
  {
    std::lock_guard<std::mutex> guard(expectedMutex);
    if (!cachedExpected ||
        cachedExpected->path != expectedPath ||
        cachedExpected->scale != outValCol.type().scale()) {
      cachedExpected = std::make_shared<ExpectedCache>(
          loadExpectedValues(expectedPath, outValCol.type().scale()));
    }
    expected = cachedExpected;
  }

  auto outRows = outputView.num_rows();
  result.outputKeys = outRows;
  result.expectedKeys = expected->values.size();
  if (batchRows <= 0) {
    batchRows = outRows;
  }

  auto outKeyElementSize = cudf::size_of(outKeyCol.type());
  auto outValElementSize = cudf::size_of(outValCol.type());
  auto const* outKeyBase = static_cast<const uint8_t*>(outKeyCol.head()) +
      outKeyCol.offset() * outKeyElementSize;
  auto const* outValBase = static_cast<const uint8_t*>(outValCol.head()) +
      outValCol.offset() * outValElementSize;

  std::vector<uint8_t> outKeyHost;
  std::vector<uint8_t> outValHost;

  for (int64_t start = 0; start < outRows; start += batchRows) {
    auto const rowsThis = std::min<int64_t>(batchRows, outRows - start);
    auto const outKeyBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(outKeyElementSize);
    auto const outValBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(outValElementSize);
    outKeyHost.resize(outKeyBytes);
    outValHost.resize(outValBytes);

    auto const* outKeyPtr = outKeyBase + start * outKeyElementSize;
    auto const* outValPtr = outValBase + start * outValElementSize;
    auto outKeyStatus = cudaMemcpyAsync(
        outKeyHost.data(),
        outKeyPtr,
        outKeyBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto outValStatus = cudaMemcpyAsync(
        outValHost.data(),
        outValPtr,
        outValBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (outKeyStatus != cudaSuccess || outValStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "output cudaMemcpyAsync failed";
      return result;
    }
    auto outSync = cudaStreamSynchronize(stream.value());
    if (outSync != cudaSuccess) {
      result.skipped = true;
      result.reason = "output cudaStreamSynchronize failed";
      return result;
    }

    for (int64_t i = 0; i < rowsThis; ++i) {
      int64_t key = 0;
      if (outKeyType == cudf::type_id::INT64) {
        std::memcpy(
            &key, outKeyHost.data() + i * outKeyElementSize, sizeof(int64_t));
      } else {
        int32_t key32 = 0;
        std::memcpy(
            &key32,
            outKeyHost.data() + i * outKeyElementSize,
            sizeof(int32_t));
        key = key32;
      }
      __int128_t actual = 0;
      if (outValType == cudf::type_id::DECIMAL64) {
        int64_t value64 = 0;
        std::memcpy(
            &value64,
            outValHost.data() + i * outValElementSize,
            sizeof(int64_t));
        actual = static_cast<__int128_t>(value64);
      } else {
        uint64_t lo = 0;
        int64_t hi = 0;
        auto const* ptr = outValHost.data() + i * outValElementSize;
        std::memcpy(&lo, ptr, sizeof(uint64_t));
        std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
        actual = (static_cast<__int128_t>(hi) << 64) | lo;
      }

      auto it = expected->values.find(key);
      if (it == expected->values.end()) {
        result.missing++;
        if (result.mismatches == 0) {
          result.firstKey = key;
          result.firstExpected = 0;
          result.firstActual = actual;
        }
        result.mismatches++;
        continue;
      }
      result.checked++;
      if (it->second.value != actual) {
        if (result.mismatches == 0) {
          result.firstKey = key;
          result.firstExpected = it->second.value;
          result.firstActual = actual;
        }
        result.mismatches++;
      }
    }
  }

  return result;
}

std::optional<std::pair<size_t, size_t>> findSumAggregation(
    std::vector<cudf::groupby::aggregation_request> const& requests) {
  for (size_t reqIdx = 0; reqIdx < requests.size(); ++reqIdx) {
    auto const& aggregations = requests[reqIdx].aggregations;
    for (size_t aggIdx = 0; aggIdx < aggregations.size(); ++aggIdx) {
      if (aggregations[aggIdx]->kind == cudf::aggregation::Kind::SUM) {
        return std::make_pair(reqIdx, aggIdx);
      }
    }
  }
  return std::nullopt;
}

EndToEndValidationResult validateEndToEndHashAgg(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    cudf::null_policy nullPolicy,
    std::vector<cudf::groupby::aggregation_request> const& requests,
    cudf::table_view const& outputView,
    int64_t maxRows,
    int64_t batchRows) {
  EndToEndValidationResult result;
  if (groupbyKeyView.num_columns() != 1) {
    result.skipped = true;
    result.reason = "grouping key count != 1";
    return result;
  }
  if (outputView.num_columns() != 2) {
    result.skipped = true;
    result.reason = "output column count != keys+1";
    return result;
  }
  auto sumInfo = findSumAggregation(requests);
  if (!sumInfo.has_value()) {
    result.skipped = true;
    result.reason = "no SUM aggregation";
    return result;
  }

  auto const& keyCol = groupbyKeyView.column(0);
  auto const& valueCol = requests[sumInfo->first].values;
  if (!cudf::is_fixed_width(keyCol.type()) ||
      !cudf::is_fixed_width(valueCol.type())) {
    result.skipped = true;
    result.reason = "non-fixed-width input";
    return result;
  }

  auto keyType = keyCol.type().id();
  auto valueType = valueCol.type().id();
  if (keyType != cudf::type_id::INT64 &&
      keyType != cudf::type_id::INT32) {
    result.skipped = true;
    result.reason = "unsupported key type";
    return result;
  }
  if (valueType != cudf::type_id::DECIMAL64 &&
      valueType != cudf::type_id::DECIMAL128) {
    result.skipped = true;
    result.reason = "unsupported value type";
    return result;
  }

  int64_t totalRows = groupbyKeyView.num_rows();
  if (maxRows > 0 && totalRows > maxRows) {
    totalRows = maxRows;
  }
  if (batchRows <= 0) {
    batchRows = totalRows;
  }

  struct ExpectedAgg {
    __int128_t sum{0};
    int64_t countNonNull{0};
    int64_t rows{0};
    bool seen{false};
  };
  std::unordered_map<int64_t, ExpectedAgg> expected;
  expected.reserve(static_cast<size_t>(std::min<int64_t>(totalRows, 1000000)));
  ExpectedAgg nullKeyAgg;
  bool sawNullKey = false;

  auto keyElementSize = cudf::size_of(keyCol.type());
  auto valueElementSize = cudf::size_of(valueCol.type());
  auto const* keyBase = static_cast<const uint8_t*>(keyCol.head()) +
      keyCol.offset() * keyElementSize;
  auto const* valueBase = static_cast<const uint8_t*>(valueCol.head()) +
      valueCol.offset() * valueElementSize;
  auto keyBitOffset = keyCol.offset();
  auto valueBitOffset = valueCol.offset();

  std::vector<uint8_t> keyMask;
  std::vector<uint8_t> valueMask;
  if (keyCol.null_count() > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(keyCol.size()));
    keyMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        keyMask.data(),
        keyCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "key null mask copy failed";
      return result;
    }
  }
  if (valueCol.null_count() > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(valueCol.size()));
    valueMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        valueMask.data(),
        valueCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "value null mask copy failed";
      return result;
    }
  }
  if (!keyMask.empty() || !valueMask.empty()) {
    auto const syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "null mask stream sync failed";
      return result;
    }
  }

  auto isValid = [](const std::vector<uint8_t>& mask, int64_t index) -> bool {
    if (mask.empty()) {
      return true;
    }
    auto const byte = mask[static_cast<size_t>(index) / 8];
    return ((byte >> (index % 8)) & 1) != 0;
  };

  std::vector<uint8_t> keyHost;
  std::vector<uint8_t> valueHost;
  for (int64_t start = 0; start < totalRows; start += batchRows) {
    auto const rowsThis = std::min<int64_t>(batchRows, totalRows - start);
    auto const keyBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(keyElementSize);
    auto const valueBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(valueElementSize);
    keyHost.resize(keyBytes);
    valueHost.resize(valueBytes);

    auto const* keyPtr = keyBase + start * keyElementSize;
    auto const* valuePtr = valueBase + start * valueElementSize;
    auto keyStatus = cudaMemcpyAsync(
        keyHost.data(),
        keyPtr,
        keyBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto valueStatus = cudaMemcpyAsync(
        valueHost.data(),
        valuePtr,
        valueBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (keyStatus != cudaSuccess || valueStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "cudaMemcpyAsync failed";
      return result;
    }
    auto syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "cudaStreamSynchronize failed";
      return result;
    }

    for (int64_t i = 0; i < rowsThis; ++i) {
      auto const rowIndex = start + i;
      bool keyValid = isValid(keyMask, keyBitOffset + rowIndex);
      bool valueValid = isValid(valueMask, valueBitOffset + rowIndex);
      int64_t key = 0;
      if (keyValid) {
        if (keyType == cudf::type_id::INT64) {
          std::memcpy(
              &key, keyHost.data() + i * keyElementSize, sizeof(int64_t));
        } else {
          int32_t key32 = 0;
          std::memcpy(
              &key32, keyHost.data() + i * keyElementSize, sizeof(int32_t));
          key = key32;
        }
      }

      if (!keyValid && nullPolicy == cudf::null_policy::EXCLUDE) {
        continue;
      }

      ExpectedAgg* agg = nullptr;
      if (!keyValid) {
        sawNullKey = true;
        agg = &nullKeyAgg;
      } else {
        agg = &expected[key];
      }
      agg->rows += 1;
      if (!valueValid) {
        continue;
      }
      __int128_t value = 0;
      if (valueType == cudf::type_id::DECIMAL64) {
        int64_t value64 = 0;
        std::memcpy(
            &value64,
            valueHost.data() + i * valueElementSize,
            sizeof(int64_t));
        value = static_cast<__int128_t>(value64);
      } else {
        uint64_t lo = 0;
        int64_t hi = 0;
        auto const* ptr = valueHost.data() + i * valueElementSize;
        std::memcpy(&lo, ptr, sizeof(uint64_t));
        std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
        value = (static_cast<__int128_t>(hi) << 64) | lo;
      }
      agg->sum += value;
      agg->countNonNull += 1;
    }
  }

  auto outKeyCol = outputView.column(0);
  auto outValCol = outputView.column(1);
  if (!cudf::is_fixed_width(outKeyCol.type()) ||
      !cudf::is_fixed_width(outValCol.type())) {
    result.skipped = true;
    result.reason = "non-fixed-width output";
    return result;
  }

  auto outKeyType = outKeyCol.type().id();
  auto outValType = outValCol.type().id();
  if (outKeyType != keyType || outValType != valueType) {
    result.skipped = true;
    result.reason = "output type mismatch";
    return result;
  }

  auto outRows = outputView.num_rows();
  result.outputKeys = outRows;
  result.expectedKeys =
      static_cast<int64_t>(expected.size()) + (sawNullKey ? 1 : 0);

  auto outKeyElementSize = cudf::size_of(outKeyCol.type());
  auto outValElementSize = cudf::size_of(outValCol.type());
  auto const* outKeyBase = static_cast<const uint8_t*>(outKeyCol.head()) +
      outKeyCol.offset() * outKeyElementSize;
  auto const* outValBase = static_cast<const uint8_t*>(outValCol.head()) +
      outValCol.offset() * outValElementSize;
  auto outKeyBitOffset = outKeyCol.offset();
  auto outValBitOffset = outValCol.offset();

  std::vector<uint8_t> outKeyMask;
  std::vector<uint8_t> outValMask;
  if (outKeyCol.null_count() > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(outKeyCol.size()));
    outKeyMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        outKeyMask.data(),
        outKeyCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "output key mask copy failed";
      return result;
    }
  }
  if (outValCol.null_count() > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(outValCol.size()));
    outValMask.resize(maskBytes);
    auto const copyStatus = cudaMemcpyAsync(
        outValMask.data(),
        outValCol.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "output value mask copy failed";
      return result;
    }
  }
  if (!outKeyMask.empty() || !outValMask.empty()) {
    auto const syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "output mask stream sync failed";
      return result;
    }
  }

  std::vector<uint8_t> outKeyHost;
  std::vector<uint8_t> outValHost;
  for (int64_t start = 0; start < outRows; start += batchRows) {
    auto const rowsThis = std::min<int64_t>(batchRows, outRows - start);
    auto const outKeyBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(outKeyElementSize);
    auto const outValBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(outValElementSize);
    outKeyHost.resize(outKeyBytes);
    outValHost.resize(outValBytes);

    auto const* outKeyPtr = outKeyBase + start * outKeyElementSize;
    auto const* outValPtr = outValBase + start * outValElementSize;
    auto outKeyStatus = cudaMemcpyAsync(
        outKeyHost.data(),
        outKeyPtr,
        outKeyBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto outValStatus = cudaMemcpyAsync(
        outValHost.data(),
        outValPtr,
        outValBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (outKeyStatus != cudaSuccess || outValStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "output cudaMemcpyAsync failed";
      return result;
    }
    auto outSync = cudaStreamSynchronize(stream.value());
    if (outSync != cudaSuccess) {
      result.skipped = true;
      result.reason = "output cudaStreamSynchronize failed";
      return result;
    }

    for (int64_t i = 0; i < rowsThis; ++i) {
      auto const rowIndex = start + i;
      bool keyValid = isValid(outKeyMask, outKeyBitOffset + rowIndex);
      bool valueValid = isValid(outValMask, outValBitOffset + rowIndex);
      int64_t key = 0;
      if (keyValid) {
        if (outKeyType == cudf::type_id::INT64) {
          std::memcpy(
              &key, outKeyHost.data() + i * outKeyElementSize, sizeof(int64_t));
        } else {
          int32_t key32 = 0;
          std::memcpy(
              &key32,
              outKeyHost.data() + i * outKeyElementSize,
              sizeof(int32_t));
          key = key32;
        }
      }
      __int128_t actual = 0;
      if (valueValid) {
        if (outValType == cudf::type_id::DECIMAL64) {
          int64_t value64 = 0;
          std::memcpy(
              &value64,
              outValHost.data() + i * outValElementSize,
              sizeof(int64_t));
          actual = static_cast<__int128_t>(value64);
        } else {
          uint64_t lo = 0;
          int64_t hi = 0;
          auto const* ptr = outValHost.data() + i * outValElementSize;
          std::memcpy(&lo, ptr, sizeof(uint64_t));
          std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
          actual = (static_cast<__int128_t>(hi) << 64) | lo;
        }
      }

      ExpectedAgg* agg = nullptr;
      if (!keyValid) {
        if (!sawNullKey) {
          result.missing++;
          if (result.mismatches == 0) {
            result.firstKey = 0;
            result.firstExpected = 0;
            result.firstActual = actual;
          }
          result.mismatches++;
          continue;
        }
        agg = &nullKeyAgg;
      } else {
        auto it = expected.find(key);
        if (it == expected.end()) {
          result.missing++;
          if (result.mismatches == 0) {
            result.firstKey = key;
            result.firstExpected = 0;
            result.firstActual = actual;
          }
          result.mismatches++;
          continue;
        }
        agg = &it->second;
      }

      agg->seen = true;
      result.checked++;
      if (!valueValid) {
        if (agg->countNonNull != 0) {
          if (result.mismatches == 0) {
            result.firstKey = key;
            result.firstExpected = agg->sum;
            result.firstActual = 0;
          }
          result.mismatches++;
        }
        continue;
      }
      if (agg->countNonNull == 0 || agg->sum != actual) {
        if (result.mismatches == 0) {
          result.firstKey = key;
          result.firstExpected = agg->sum;
          result.firstActual = actual;
        }
        result.mismatches++;
      }
    }
  }

  for (auto& entry : expected) {
    if (!entry.second.seen && entry.second.rows > 0) {
      result.missing++;
    }
  }
  if (sawNullKey && !nullKeyAgg.seen && nullKeyAgg.rows > 0) {
    result.missing++;
  }
  return result;
}

struct StateRoundtripResult {
  bool skipped{false};
  std::string reason;
  int64_t rows{0};
  int64_t checked{0};
  int64_t maskMismatches{0};
  int64_t sumMismatches{0};
  int64_t countMismatches{0};
  int64_t firstRow{-1};
  __int128_t expectedSum{0};
  __int128_t actualSum{0};
  int64_t expectedCount{0};
  int64_t actualCount{0};
  bool expectedValid{false};
  bool actualValid{false};
};

StateRoundtripResult validateDecimalStateRoundtrip(
    cudf::column_view const& sumCol,
    cudf::column_view const& countCol,
    cudf::column_view const& stateCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    int64_t maxRows,
    int64_t batchRows) {
  StateRoundtripResult result;
  if (stateCol.type().id() != cudf::type_id::STRING) {
    result.skipped = true;
    result.reason = "state type not string";
    return result;
  }
  if (countCol.type().id() != cudf::type_id::INT64) {
    result.skipped = true;
    result.reason = "count type not int64";
    return result;
  }
  auto sumType = sumCol.type().id();
  if (sumType != cudf::type_id::DECIMAL64 &&
      sumType != cudf::type_id::DECIMAL128) {
    result.skipped = true;
    result.reason = "sum type not decimal";
    return result;
  }
  if (sumCol.size() != countCol.size() ||
      sumCol.size() != stateCol.size()) {
    result.skipped = true;
    result.reason = "column size mismatch";
    return result;
  }

  auto decoded = cudf_velox::deserializeDecimalSumStateWithCount(
      stateCol, scale, stream);
  auto decodedSumView = decoded.sum->view();
  auto decodedCountView = decoded.count->view();
  if (decodedSumView.size() != stateCol.size() ||
      decodedCountView.size() != stateCol.size()) {
    result.skipped = true;
    result.reason = "decoded size mismatch";
    return result;
  }
  if (decodedCountView.type().id() != cudf::type_id::INT64) {
    result.skipped = true;
    result.reason = "decoded count type mismatch";
    return result;
  }

  int64_t totalRows = stateCol.size();
  if (maxRows > 0 && totalRows > maxRows) {
    totalRows = maxRows;
  }
  result.rows = totalRows;
  if (batchRows <= 0) {
    batchRows = totalRows;
  }

  std::vector<uint8_t> stateMask;
  std::vector<uint8_t> sumMask;
  std::vector<uint8_t> countMask;
  auto stateBitOffset = stateCol.offset();
  auto sumBitOffset = sumCol.offset();
  auto countBitOffset = countCol.offset();

  auto copyMask = [&](cudf::column_view const& col,
                      std::vector<uint8_t>& mask) -> bool {
    if (col.null_count() == 0) {
      return true;
    }
    auto const maskBytes = static_cast<size_t>(
        cudf::bitmask_allocation_size_bytes(col.offset() + col.size()));
    mask.resize(maskBytes);
    auto const status = cudaMemcpyAsync(
        mask.data(),
        col.null_mask(),
        maskBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    return status == cudaSuccess;
  };

  if (!copyMask(stateCol, stateMask)) {
    result.skipped = true;
    result.reason = "state mask copy failed";
    return result;
  }
  if (!copyMask(sumCol, sumMask)) {
    result.skipped = true;
    result.reason = "sum mask copy failed";
    return result;
  }
  if (!copyMask(countCol, countMask)) {
    result.skipped = true;
    result.reason = "count mask copy failed";
    return result;
  }
  if (!stateMask.empty() || !sumMask.empty() || !countMask.empty()) {
    auto const syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "mask stream sync failed";
      return result;
    }
  }

  auto isValid = [](const std::vector<uint8_t>& mask, int64_t index) -> bool {
    if (mask.empty()) {
      return true;
    }
    auto const byte = mask[static_cast<size_t>(index) / 8];
    return ((byte >> (index % 8)) & 1) != 0;
  };

  auto sumElementSize = cudf::size_of(sumCol.type());
  auto countElementSize = cudf::size_of(countCol.type());
  auto decodedSumElementSize = cudf::size_of(decodedSumView.type());
  auto decodedCountElementSize = cudf::size_of(decodedCountView.type());

  auto const* sumBase = static_cast<const uint8_t*>(sumCol.head()) +
      sumCol.offset() * sumElementSize;
  auto const* countBase = static_cast<const uint8_t*>(countCol.head()) +
      countCol.offset() * countElementSize;
  auto const* decodedSumBase =
      static_cast<const uint8_t*>(decodedSumView.head()) +
      decodedSumView.offset() * decodedSumElementSize;
  auto const* decodedCountBase =
      static_cast<const uint8_t*>(decodedCountView.head()) +
      decodedCountView.offset() * decodedCountElementSize;

  auto readSumValue = [&](const std::vector<uint8_t>& buffer,
                          int64_t index,
                          cudf::type_id type) -> __int128_t {
    if (type == cudf::type_id::DECIMAL64) {
      int64_t value64 = 0;
      std::memcpy(
          &value64,
          buffer.data() + index * sumElementSize,
          sizeof(int64_t));
      return static_cast<__int128_t>(value64);
    }
    uint64_t lo = 0;
    int64_t hi = 0;
    auto const* ptr = buffer.data() + index * sumElementSize;
    std::memcpy(&lo, ptr, sizeof(uint64_t));
    std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
    return (static_cast<__int128_t>(hi) << 64) | lo;
  };

  auto readDecodedSumValue = [&](const std::vector<uint8_t>& buffer,
                                 int64_t index) -> __int128_t {
    uint64_t lo = 0;
    int64_t hi = 0;
    auto const* ptr = buffer.data() + index * decodedSumElementSize;
    std::memcpy(&lo, ptr, sizeof(uint64_t));
    std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
    return (static_cast<__int128_t>(hi) << 64) | lo;
  };

  auto readCountValue = [&](const std::vector<uint8_t>& buffer,
                            int64_t index) -> int64_t {
    int64_t value = 0;
    std::memcpy(
        &value,
        buffer.data() + index * countElementSize,
        sizeof(int64_t));
    return value;
  };

  auto readDecodedCountValue = [&](const std::vector<uint8_t>& buffer,
                                   int64_t index) -> int64_t {
    int64_t value = 0;
    std::memcpy(
        &value,
        buffer.data() + index * decodedCountElementSize,
        sizeof(int64_t));
    return value;
  };

  std::vector<uint8_t> sumHost;
  std::vector<uint8_t> countHost;
  std::vector<uint8_t> decodedSumHost;
  std::vector<uint8_t> decodedCountHost;

  for (int64_t start = 0; start < totalRows; start += batchRows) {
    auto const rowsThis = std::min<int64_t>(batchRows, totalRows - start);
    auto const sumBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(sumElementSize);
    auto const countBytes =
        static_cast<size_t>(rowsThis) * static_cast<size_t>(countElementSize);
    auto const decodedSumBytes = static_cast<size_t>(rowsThis) *
        static_cast<size_t>(decodedSumElementSize);
    auto const decodedCountBytes = static_cast<size_t>(rowsThis) *
        static_cast<size_t>(decodedCountElementSize);
    sumHost.resize(sumBytes);
    countHost.resize(countBytes);
    decodedSumHost.resize(decodedSumBytes);
    decodedCountHost.resize(decodedCountBytes);

    auto const* sumPtr = sumBase + start * sumElementSize;
    auto const* countPtr = countBase + start * countElementSize;
    auto const* decodedSumPtr = decodedSumBase + start * decodedSumElementSize;
    auto const* decodedCountPtr =
        decodedCountBase + start * decodedCountElementSize;

    auto sumStatus = cudaMemcpyAsync(
        sumHost.data(),
        sumPtr,
        sumBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto countStatus = cudaMemcpyAsync(
        countHost.data(),
        countPtr,
        countBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto decodedSumStatus = cudaMemcpyAsync(
        decodedSumHost.data(),
        decodedSumPtr,
        decodedSumBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    auto decodedCountStatus = cudaMemcpyAsync(
        decodedCountHost.data(),
        decodedCountPtr,
        decodedCountBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (sumStatus != cudaSuccess || countStatus != cudaSuccess ||
        decodedSumStatus != cudaSuccess || decodedCountStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "cudaMemcpyAsync failed";
      return result;
    }
    auto syncStatus = cudaStreamSynchronize(stream.value());
    if (syncStatus != cudaSuccess) {
      result.skipped = true;
      result.reason = "cudaStreamSynchronize failed";
      return result;
    }

    for (int64_t i = 0; i < rowsThis; ++i) {
      auto const rowIndex = start + i;
      bool stateValid = isValid(stateMask, stateBitOffset + rowIndex);
      bool sumValid = isValid(sumMask, sumBitOffset + rowIndex);
      bool countValid = isValid(countMask, countBitOffset + rowIndex);
      int64_t countValue = readCountValue(countHost, i);
      bool expectedValid =
          sumValid && countValid && static_cast<int64_t>(countValue) != 0;
      if (stateValid != expectedValid) {
        result.maskMismatches++;
        if (result.firstRow < 0) {
          result.firstRow = rowIndex;
          result.expectedValid = expectedValid;
          result.actualValid = stateValid;
          result.expectedCount = countValue;
        }
      }
      if (!stateValid) {
        continue;
      }
      result.checked++;
      __int128_t expectedSum = readSumValue(sumHost, i, sumType);
      __int128_t actualSum = readDecodedSumValue(decodedSumHost, i);
      if (expectedSum != actualSum) {
        result.sumMismatches++;
        if (result.firstRow < 0) {
          result.firstRow = rowIndex;
          result.expectedSum = expectedSum;
          result.actualSum = actualSum;
          result.expectedCount = countValue;
          result.actualCount = readDecodedCountValue(decodedCountHost, i);
          result.expectedValid = expectedValid;
          result.actualValid = stateValid;
        }
      }
      int64_t actualCount = readDecodedCountValue(decodedCountHost, i);
      if (countValue != actualCount) {
        result.countMismatches++;
        if (result.firstRow < 0) {
          result.firstRow = rowIndex;
          result.expectedSum = expectedSum;
          result.actualSum = actualSum;
          result.expectedCount = countValue;
          result.actualCount = actualCount;
          result.expectedValid = expectedValid;
          result.actualValid = stateValid;
        }
      }
    }
  }

  return result;
}

const char* fakeGroupbyModeName(int32_t mode) {
  switch (mode) {
    case 1:
      return "zero";
    case 2:
      return "one";
    default:
      return "disabled";
  }
}

std::unique_ptr<cudf::scalar> makeFakeConstantScalar(
    cudf::data_type type,
    int32_t mode,
    rmm::cuda_stream_view stream) {
  if (mode == 2) {
    switch (type.id()) {
      case cudf::type_id::INT8:
        return cudf::make_fixed_width_scalar<int8_t>(1, stream);
      case cudf::type_id::INT16:
        return cudf::make_fixed_width_scalar<int16_t>(1, stream);
      case cudf::type_id::INT32:
        return cudf::make_fixed_width_scalar<int32_t>(1, stream);
      case cudf::type_id::INT64:
        return cudf::make_fixed_width_scalar<int64_t>(1, stream);
      case cudf::type_id::UINT8:
        return cudf::make_fixed_width_scalar<uint8_t>(1, stream);
      case cudf::type_id::UINT16:
        return cudf::make_fixed_width_scalar<uint16_t>(1, stream);
      case cudf::type_id::UINT32:
        return cudf::make_fixed_width_scalar<uint32_t>(1, stream);
      case cudf::type_id::UINT64:
        return cudf::make_fixed_width_scalar<uint64_t>(1, stream);
      case cudf::type_id::FLOAT32:
        return cudf::make_fixed_width_scalar<float>(1.0f, stream);
      case cudf::type_id::FLOAT64:
        return cudf::make_fixed_width_scalar<double>(1.0, stream);
      case cudf::type_id::BOOL8:
        return cudf::make_fixed_width_scalar<bool>(true, stream);
      default:
        break;
    }
  }
  auto scalar = cudf::make_default_constructed_scalar(type, stream);
  scalar->set_valid_async(true, stream);
  return scalar;
}

std::string runGroupByAggregateRequestIsolationProbes(
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& tableView,
    cudf::table_view const& groupbyKeyView,
    cudf::null_policy nullPolicy,
    std::vector<cudf::groupby::aggregation_request> const& requests,
    int64_t numGroupingKeys) {
  auto const selectedGroup = hashAggDebugDeviceSyncPoint();
  if (!isDeviceSyncGroupEnabled(
          selectedGroup, kDeviceSyncGroupGroupByRequestIsolation)) {
    return "";
  }

  bool exhaustive = false;
  auto const requestSubsets = buildRequestProbeSubsets(requests.size(), exhaustive);
  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "aggregateProbe.begin step="
            << stepName(step) << " stream="
            << reinterpret_cast<const void*>(stream.value()) << " rows="
            << tableView.num_rows() << " cols=" << tableView.num_columns()
            << " requestCount=" << requests.size()
            << " subsetCount=" << requestSubsets.size()
            << " exhaustive=" << (exhaustive ? 1 : 0)
            << " selectedGroup=" << selectedGroup
            << " probeMode=copyThenView";

  for (size_t probeOrdinal = 0; probeOrdinal < requestSubsets.size();
       ++probeOrdinal) {
    auto const& requestSubset = requestSubsets[probeOrdinal];
    auto const subsetLabel = formatRequestSubset(requestSubset);
    auto const probeIdBase = static_cast<int32_t>(1000 + probeOrdinal * 10);
    maybeDeviceSyncProbe(
        kDeviceSyncGroupGroupByRequestIsolation,
        probeIdBase + 1,
        "HashAgg.doGroupByAggregation.aggregateProbe.beforeSubset",
        step,
        stream,
        tableView.num_rows(),
        static_cast<int64_t>(requestSubset.size()),
        numGroupingKeys);
    auto const preProbePeekErr = cudaPeekAtLastError();
    auto const preProbeStreamSyncErr = cudaStreamSynchronize(stream.value());
    auto const preProbeDeviceSyncErr = cudaDeviceSynchronize();
    LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                 "aggregateProbe.preState step="
              << stepName(step) << " stream="
              << reinterpret_cast<const void*>(stream.value())
              << " probeOrdinal=" << probeOrdinal << " subset=" << subsetLabel
              << " cudaPeek=" << cudaErrorNameSafe(preProbePeekErr) << "("
              << static_cast<int>(preProbePeekErr) << ")"
              << " cudaPeekMsg=" << cudaErrorStringSafe(preProbePeekErr)
              << " cudaStreamSync="
              << cudaErrorNameSafe(preProbeStreamSyncErr) << "("
              << static_cast<int>(preProbeStreamSyncErr) << ")"
              << " cudaStreamSyncMsg="
              << cudaErrorStringSafe(preProbeStreamSyncErr)
              << " cudaDeviceSync="
              << cudaErrorNameSafe(preProbeDeviceSyncErr) << "("
              << static_cast<int>(preProbeDeviceSyncErr) << ")"
              << " cudaDeviceSyncMsg="
              << cudaErrorStringSafe(preProbeDeviceSyncErr);
    if (preProbePeekErr != cudaSuccess ||
        preProbeStreamSyncErr != cudaSuccess ||
        preProbeDeviceSyncErr != cudaSuccess) {
      return "probeOrdinal=" + std::to_string(probeOrdinal) + " subset=" +
          subsetLabel + " preState cudaPeek=" +
          std::string(cudaErrorNameSafe(preProbePeekErr)) + "(" +
          std::to_string(static_cast<int>(preProbePeekErr)) + ")" +
          " cudaStreamSync=" +
          std::string(cudaErrorNameSafe(preProbeStreamSyncErr)) + "(" +
          std::to_string(static_cast<int>(preProbeStreamSyncErr)) + ")" +
          " cudaDeviceSync=" +
          std::string(cudaErrorNameSafe(preProbeDeviceSyncErr)) + "(" +
          std::to_string(static_cast<int>(preProbeDeviceSyncErr)) + ")";
    }

    std::vector<cudf::groupby::aggregation_request> viewProbeRequests;
    try {
      viewProbeRequests = cloneGroupbyRequestSubset(requests, requestSubset);
    } catch (const std::exception& e) {
      return "probeOrdinal=" + std::to_string(probeOrdinal) + " subset=" +
          subsetLabel + " viewCloneException=" + e.what();
    }

    CopiedProbeRequests copiedProbeRequests;
    try {
      copiedProbeRequests =
          buildCopiedGroupbyRequestSubset(requests, requestSubset, stream);
    } catch (const std::exception& e) {
      return "probeOrdinal=" + std::to_string(probeOrdinal) + " subset=" +
          subsetLabel + " copyCloneException=" + e.what();
    }

    maybeDeviceSyncProbe(
        kDeviceSyncGroupGroupByRequestIsolation,
        probeIdBase + 2,
        "HashAgg.doGroupByAggregation.aggregateProbe.beforeCopyAggregate",
        step,
        stream,
        tableView.num_rows(),
        static_cast<int64_t>(requestSubset.size()),
        numGroupingKeys);
    std::optional<AggregateProbeHostSnapshot> copyProbeSnapshot;
    if (!hashAggDebugProbeDumpDir().empty()) {
      copyProbeSnapshot = captureAggregateProbeHostSnapshot(
          stream,
          groupbyKeyView,
          copiedProbeRequests.requests,
          requestSubset,
          hashAggDebugProbeDumpDir(),
          hashAggDebugProbeDumpMaxRows());
    }
    logGroupbyRequestsDebug(step, stream, tableView, copiedProbeRequests.requests);
    auto const copyStatus = runAggregateProbeCall(
        step,
        stream,
        groupbyKeyView,
        nullPolicy,
        copiedProbeRequests.requests,
        probeOrdinal,
        subsetLabel,
        "copy");
    maybeDeviceSyncProbe(
        kDeviceSyncGroupGroupByRequestIsolation,
        probeIdBase + 3,
        "HashAgg.doGroupByAggregation.aggregateProbe.afterCopyAggregate",
        step,
        stream,
        copyStatus.outputRows >= 0 ? copyStatus.outputRows : tableView.num_rows(),
        static_cast<int64_t>(requestSubset.size()),
        copyStatus.outputCount >= 0 ? copyStatus.outputCount : -1);
    if (aggregateProbeFailed(copyStatus)) {
      auto failure = formatAggregateProbeFailure(
          probeOrdinal, subsetLabel, "copy", copyStatus);
      if (copyProbeSnapshot.has_value()) {
        auto const dumpDir = maybeWriteAggregateProbeDump(
            step,
            probeOrdinal,
            requestSubset,
            "copy",
            nullPolicy,
            copyStatus,
            *copyProbeSnapshot);
        if (!dumpDir.empty()) {
          failure += " dumpDir=" + dumpDir;
        }
      }
      return failure;
    }

    maybeDeviceSyncProbe(
        kDeviceSyncGroupGroupByRequestIsolation,
        probeIdBase + 4,
        "HashAgg.doGroupByAggregation.aggregateProbe.beforeViewAggregate",
        step,
        stream,
        tableView.num_rows(),
        static_cast<int64_t>(requestSubset.size()),
        numGroupingKeys);
    std::optional<AggregateProbeHostSnapshot> viewProbeSnapshot;
    if (!hashAggDebugProbeDumpDir().empty()) {
      viewProbeSnapshot = captureAggregateProbeHostSnapshot(
          stream,
          groupbyKeyView,
          viewProbeRequests,
          requestSubset,
          hashAggDebugProbeDumpDir(),
          hashAggDebugProbeDumpMaxRows());
    }
    logGroupbyRequestsDebug(step, stream, tableView, viewProbeRequests);
    auto const viewStatus = runAggregateProbeCall(
        step,
        stream,
        groupbyKeyView,
        nullPolicy,
        viewProbeRequests,
        probeOrdinal,
        subsetLabel,
        "view");
    maybeDeviceSyncProbe(
        kDeviceSyncGroupGroupByRequestIsolation,
        probeIdBase + 5,
        "HashAgg.doGroupByAggregation.aggregateProbe.afterViewAggregate",
        step,
        stream,
        viewStatus.outputRows >= 0 ? viewStatus.outputRows : tableView.num_rows(),
        static_cast<int64_t>(requestSubset.size()),
        viewStatus.outputCount >= 0 ? viewStatus.outputCount : -1);
    if (aggregateProbeFailed(viewStatus)) {
      auto failure = formatAggregateProbeFailure(
          probeOrdinal, subsetLabel, "view", viewStatus);
      if (viewProbeSnapshot.has_value()) {
        auto const dumpDir = maybeWriteAggregateProbeDump(
            step,
            probeOrdinal,
            requestSubset,
            "view",
            nullPolicy,
            viewStatus,
            *viewProbeSnapshot);
        if (!dumpDir.empty()) {
          failure += " dumpDir=" + dumpDir;
        }
      }
      return failure;
    }
  }

  LOG(INFO) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
               "aggregateProbe.end step="
            << stepName(step) << " stream="
            << reinterpret_cast<const void*>(stream.value()) << " requestCount="
            << requests.size() << " subsetCount=" << requestSubsets.size()
            << " exhaustive=" << (exhaustive ? 1 : 0)
            << " status=all-pass";
  return "";
}

bool hasDecimalGroupbyRequest(
    std::vector<cudf::groupby::aggregation_request> const& requests) {
  return std::any_of(requests.begin(), requests.end(), [](auto const& request) {
    auto const typeId = request.values.type().id();
    return typeId == cudf::type_id::DECIMAL64 ||
        typeId == cudf::type_id::DECIMAL128;
  });
}

bool hasDecimalAggregator(
    std::vector<std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator>> const&
        aggregators) {
  return std::any_of(aggregators.begin(), aggregators.end(), [](auto const& agg) {
    return agg && agg->resultType && agg->resultType->isDecimal();
  });
}

bool isCpuDetourSupportedAggKind(cudf::aggregation::Kind kind) {
  return kind == cudf::aggregation::SUM || kind == cudf::aggregation::COUNT_VALID ||
      kind == cudf::aggregation::COUNT_ALL;
}

std::string cpuDetourAggKindName(cudf::aggregation::Kind kind) {
  switch (kind) {
    case cudf::aggregation::SUM:
      return "SUM";
    case cudf::aggregation::COUNT_VALID:
      return "COUNT_VALID";
    case cudf::aggregation::COUNT_ALL:
      return "COUNT_ALL";
    default:
      return "kind_" + std::to_string(static_cast<int>(kind));
  }
}

std::string buildGroupKeySignature(
    RowVectorPtr const& keys,
    vector_size_t row,
    cudf::null_policy nullPolicy,
    bool& includeRow) {
  includeRow = true;
  std::string key;
  key.reserve(keys->childrenSize() * 16);
  for (auto i = 0; i < keys->childrenSize(); ++i) {
    auto const& child = keys->childAt(i);
    if (child->isNullAt(row)) {
      if (nullPolicy == cudf::null_policy::EXCLUDE) {
        includeRow = false;
        return "";
      }
      key += "N;";
      continue;
    }
    auto const value = child->toString(row);
    key += "V";
    key += std::to_string(value.size());
    key += ":";
    key += value;
    key += ";";
  }
  return key;
}

struct CpuDetourNumericValue {
  int128_t intValue{0};
  long double floatValue{0.0};
  bool isFloating{false};
};

bool readNumericLikeValue(
    DecodedVector const& vector,
    TypeKind kind,
    vector_size_t row,
    CpuDetourNumericValue& out,
    std::string& error) {
  switch (kind) {
    case TypeKind::TINYINT:
      out.intValue = vector.valueAt<int8_t>(row);
      out.isFloating = false;
      return true;
    case TypeKind::SMALLINT:
      out.intValue = vector.valueAt<int16_t>(row);
      out.isFloating = false;
      return true;
    case TypeKind::INTEGER:
      out.intValue = vector.valueAt<int32_t>(row);
      out.isFloating = false;
      return true;
    case TypeKind::BIGINT:
      out.intValue = vector.valueAt<int64_t>(row);
      out.isFloating = false;
      return true;
    case TypeKind::HUGEINT:
      out.intValue = vector.valueAt<int128_t>(row);
      out.isFloating = false;
      return true;
    case TypeKind::REAL:
      out.floatValue = vector.valueAt<float>(row);
      out.isFloating = true;
      return true;
    case TypeKind::DOUBLE:
      out.floatValue = vector.valueAt<double>(row);
      out.isFloating = true;
      return true;
    default:
      error = "unsupported request value type for CPU detour: " +
          std::string(TypeKindName::toName(kind));
      return false;
  }
}

struct CpuDetourAggState {
  int128_t sumInt{0};
  long double sumFloat{0.0};
  int64_t count{0};
  bool sumSeen{false};
};

VectorPtr makeCpuDetourSumVector(
    TypePtr const& type,
    std::vector<CpuDetourAggState> const& states,
    memory::MemoryPool* pool,
    std::string& error) {
  auto out = BaseVector::create(type, states.size(), pool);
  switch (type->kind()) {
    case TypeKind::TINYINT: {
      auto flat = out->asFlatVector<int8_t>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<int8_t>(states[i].sumInt));
        }
      }
      return out;
    }
    case TypeKind::SMALLINT: {
      auto flat = out->asFlatVector<int16_t>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<int16_t>(states[i].sumInt));
        }
      }
      return out;
    }
    case TypeKind::INTEGER: {
      auto flat = out->asFlatVector<int32_t>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<int32_t>(states[i].sumInt));
        }
      }
      return out;
    }
    case TypeKind::BIGINT: {
      auto flat = out->asFlatVector<int64_t>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<int64_t>(states[i].sumInt));
        }
      }
      return out;
    }
    case TypeKind::HUGEINT: {
      auto flat = out->asFlatVector<int128_t>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, states[i].sumInt);
        }
      }
      return out;
    }
    case TypeKind::REAL: {
      auto flat = out->asFlatVector<float>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<float>(states[i].sumFloat));
        }
      }
      return out;
    }
    case TypeKind::DOUBLE: {
      auto flat = out->asFlatVector<double>();
      for (auto i = 0; i < states.size(); ++i) {
        if (!states[i].sumSeen) {
          flat->setNull(i, true);
        } else {
          flat->set(i, static_cast<double>(states[i].sumFloat));
        }
      }
      return out;
    }
    default:
      error =
          "unsupported SUM output type for CPU detour: " + type->toString();
      return nullptr;
  }
}

VectorPtr makeCpuDetourCountVector(
    std::vector<CpuDetourAggState> const& states,
    memory::MemoryPool* pool) {
  auto out = BaseVector::create(BIGINT(), states.size(), pool);
  auto flat = out->asFlatVector<int64_t>();
  for (auto i = 0; i < states.size(); ++i) {
    flat->set(i, states[i].count);
  }
  return out;
}

bool runDecimalCpuAggregateDetour(
    memory::MemoryPool* pool,
    core::AggregationNode::Step step,
    rmm::cuda_stream_view stream,
    cudf::table_view const& groupbyKeyView,
    cudf::null_policy nullPolicy,
    std::vector<cudf::groupby::aggregation_request> const& requests,
    std::unique_ptr<cudf::table>& groupKeysOut,
    std::vector<cudf::groupby::aggregation_result>& resultsOut,
    std::string& error) {
  if (requests.empty()) {
    error = "CPU detour requires at least one aggregation request";
    return false;
  }

  auto const hasGroupingKeys = groupbyKeyView.num_columns() > 0;
  RowVectorPtr keyRow;
  if (hasGroupingKeys) {
    keyRow = cudf_velox::with_arrow::toVeloxColumn(
        groupbyKeyView, pool, "cpu_detour_key_", stream);
  }
  std::vector<cudf::column_view> requestValues;
  requestValues.reserve(requests.size());
  for (auto const& request : requests) {
    requestValues.push_back(request.values);
  }
  cudf::table_view valueTable(requestValues);
  auto valueRow = cudf_velox::with_arrow::toVeloxColumn(
      valueTable, pool, "cpu_detour_val_", stream);

  auto const inputRows = valueRow->size();
  if (hasGroupingKeys && keyRow->size() != inputRows) {
    error = "CPU detour input size mismatch: keyRows=" +
        std::to_string(keyRow->size()) + " valueRows=" +
        std::to_string(inputRows);
    return false;
  }

  for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
    auto const valueType = valueRow->childAt(requestIdx)->type()->kind();
    if (valueType != TypeKind::BIGINT && valueType != TypeKind::HUGEINT &&
        valueType != TypeKind::INTEGER && valueType != TypeKind::SMALLINT &&
        valueType != TypeKind::TINYINT && valueType != TypeKind::REAL &&
        valueType != TypeKind::DOUBLE) {
      error = "CPU detour unsupported request value type at request[" +
          std::to_string(requestIdx) +
          "]: " + valueRow->childAt(requestIdx)->type()->toString();
      return false;
    }
    for (auto aggIdx = 0; aggIdx < requests[requestIdx].aggregations.size();
         ++aggIdx) {
      auto const kind = requests[requestIdx].aggregations[aggIdx]->kind;
      if (!isCpuDetourSupportedAggKind(kind)) {
        error = "CPU detour unsupported aggregation kind at request[" +
            std::to_string(requestIdx) + "] agg[" + std::to_string(aggIdx) +
            "]: " + cpuDetourAggKindName(kind);
        return false;
      }
    }
  }
  SelectivityVector allRows(inputRows, true);
  std::vector<std::unique_ptr<DecodedVector>> decodedValues;
  decodedValues.reserve(requests.size());
  for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
    decodedValues.push_back(std::make_unique<DecodedVector>(
        *valueRow->childAt(requestIdx), allRows));
  }

  std::unordered_map<std::string, vector_size_t> keyToGroup;
  std::vector<vector_size_t> representativeRows;
  representativeRows.reserve(std::min<vector_size_t>(inputRows, 1024));
  std::vector<std::vector<std::vector<CpuDetourAggState>>> states(
      requests.size());
  for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
    states[requestIdx].resize(requests[requestIdx].aggregations.size());
  }

  for (vector_size_t row = 0; row < inputRows; ++row) {
    bool includeRow = true;
    std::string key;
    if (hasGroupingKeys) {
      key = buildGroupKeySignature(keyRow, row, nullPolicy, includeRow);
      if (!includeRow) {
        continue;
      }
    } else {
      // Global aggregation: all rows map to one synthetic key.
      key = "__global__";
    }

    auto [it, inserted] = keyToGroup.emplace(key, representativeRows.size());
    auto const groupIdx = inserted ? representativeRows.size() : it->second;
    if (inserted) {
      representativeRows.push_back(row);
      for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
        for (auto aggIdx = 0; aggIdx < requests[requestIdx].aggregations.size();
             ++aggIdx) {
          states[requestIdx][aggIdx].emplace_back();
        }
      }
    }

    for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
      auto const valueKind = valueRow->childAt(requestIdx)->type()->kind();
      auto const& decoded = *decodedValues[requestIdx];
      auto const isNull = decoded.isNullAt(row);
      CpuDetourNumericValue value;
      if (!isNull &&
          !readNumericLikeValue(decoded, valueKind, row, value, error)) {
        return false;
      }
      for (auto aggIdx = 0; aggIdx < requests[requestIdx].aggregations.size();
           ++aggIdx) {
        auto const kind = requests[requestIdx].aggregations[aggIdx]->kind;
        auto& state = states[requestIdx][aggIdx][groupIdx];
        if (kind == cudf::aggregation::SUM) {
          if (!isNull) {
            if (value.isFloating) {
              state.sumFloat += value.floatValue;
            } else {
              state.sumInt += value.intValue;
            }
            state.sumSeen = true;
          }
        } else if (kind == cudf::aggregation::COUNT_ALL) {
          ++state.count;
        } else {
          // COUNT_VALID
          if (!isNull) {
            ++state.count;
          }
        }
      }
    }
  }

  auto const numGroups = representativeRows.size();
  if (hasGroupingKeys) {
    auto groupingIndices = allocateIndices(numGroups, pool);
    auto rawGroupingIndices = groupingIndices->asMutable<vector_size_t>();
    for (auto i = 0; i < numGroups; ++i) {
      rawGroupingIndices[i] = representativeRows[i];
    }

    std::vector<VectorPtr> keyColumns;
    keyColumns.reserve(keyRow->childrenSize());
    for (auto const& child : keyRow->children()) {
      keyColumns.push_back(BaseVector::wrapInDictionary(
          nullptr, groupingIndices, numGroups, child));
    }
    auto keyOutput = std::make_shared<RowVector>(
        pool,
        std::dynamic_pointer_cast<const RowType>(keyRow->type()),
        nullptr,
        numGroups,
        std::move(keyColumns));
    groupKeysOut = cudf_velox::with_arrow::toCudfTable(keyOutput, pool, stream);
  } else {
    // No grouping keys: group-keys table has zero columns.
    std::vector<std::unique_ptr<cudf::column>> emptyColumns;
    groupKeysOut = std::make_unique<cudf::table>(std::move(emptyColumns));
  }

  std::vector<VectorPtr> hostAggVectors;
  std::vector<std::string> hostAggNames;
  std::vector<TypePtr> hostAggTypes;
  hostAggVectors.reserve(requests.size());
  hostAggNames.reserve(requests.size());
  hostAggTypes.reserve(requests.size());
  for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
    for (auto aggIdx = 0; aggIdx < requests[requestIdx].aggregations.size();
         ++aggIdx) {
      auto const kind = requests[requestIdx].aggregations[aggIdx]->kind;
      VectorPtr out;
      if (kind == cudf::aggregation::SUM) {
        out = makeCpuDetourSumVector(
            valueRow->childAt(requestIdx)->type(),
            states[requestIdx][aggIdx],
            pool,
            error);
        if (!out) {
          return false;
        }
      } else {
        out = makeCpuDetourCountVector(states[requestIdx][aggIdx], pool);
      }
      hostAggTypes.push_back(out->type());
      hostAggNames.push_back(
          "req" + std::to_string(requestIdx) + "_agg" + std::to_string(aggIdx));
      hostAggVectors.push_back(std::move(out));
    }
  }

  auto hostAggRow = std::make_shared<RowVector>(
      pool,
      ROW(hostAggNames, hostAggTypes),
      nullptr,
      numGroups,
      std::move(hostAggVectors));
  auto hostAggTable = cudf_velox::with_arrow::toCudfTable(hostAggRow, pool, stream);
  auto hostAggColumns = hostAggTable->release();

  resultsOut.clear();
  resultsOut.resize(requests.size());
  auto nextColumn = 0;
  for (auto requestIdx = 0; requestIdx < requests.size(); ++requestIdx) {
    resultsOut[requestIdx].results.reserve(
        requests[requestIdx].aggregations.size());
    for (auto aggIdx = 0; aggIdx < requests[requestIdx].aggregations.size();
         ++aggIdx) {
      if (nextColumn >= hostAggColumns.size()) {
        error = "CPU detour result column underflow";
        return false;
      }
      resultsOut[requestIdx].results.push_back(
          std::move(hostAggColumns[nextColumn++]));
    }
  }
  if (nextColumn != hostAggColumns.size()) {
    error = "CPU detour result column overflow: used=" +
        std::to_string(nextColumn) +
        " total=" + std::to_string(hostAggColumns.size());
    return false;
  }

  if (hashAggDebugEnabled()) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "cpuDecimalDetour.success step="
                 << stepName(step) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " inRows="
                 << inputRows << " outRows=" << numGroups
                 << " requestCount=" << requests.size();
  }
  return true;
}


#define DEFINE_SIMPLE_AGGREGATOR(Name, name, KIND)                            \
  struct Name##Aggregator : cudf_velox::CudfHashAggregation::Aggregator {     \
    Name##Aggregator(                                                         \
        core::AggregationNode::Step step,                                     \
        uint32_t inputIndex,                                                  \
        VectorPtr constant,                                                   \
        bool is_global,                                                       \
        const TypePtr& resultType)                                            \
        : Aggregator(                                                         \
              step,                                                           \
              cudf::aggregation::KIND,                                        \
              inputIndex,                                                     \
              constant,                                                       \
              is_global,                                                      \
              resultType) {}                                                  \
                                                                              \
    void addGroupbyRequest(                                                   \
        cudf::table_view const& tbl,                                          \
        std::vector<cudf::groupby::aggregation_request>& requests) override { \
      VELOX_CHECK(                                                            \
          constant == nullptr,                                                \
          #Name "Aggregator does not yet support constant input");            \
      auto& request = requests.emplace_back();                                \
      output_idx = requests.size() - 1;                                       \
      request.values = tbl.column(inputIndex);                                \
      request.aggregations.push_back(                                         \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>());      \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> makeOutputColumn(                           \
        std::vector<cudf::groupby::aggregation_result>& results,              \
        rmm::cuda_stream_view stream) override {                              \
      auto col = std::move(results[output_idx].results[0]);                   \
      auto const cudfResType = cudf_velox::veloxToCudfDataType(resultType);   \
      if (col->type() != cudfResType) {                                       \
        col = cudf::cast(*col, cudfResType, stream);                          \
      }                                                                       \
      return col;                                                             \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> doReduce(                                   \
        cudf::table_view const& input,                                        \
        TypePtr const& outputType,                                            \
        rmm::cuda_stream_view stream) override {                              \
      auto const aggRequest =                                                 \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>();        \
      auto const cudfOutType = cudf_velox::veloxToCudfDataType(outputType);   \
      auto const resultScalar = cudf::reduce(                                 \
          input.column(inputIndex), *aggRequest, cudfOutType, stream);        \
      return cudf::make_column_from_scalar(*resultScalar, 1, stream);         \
    }                                                                         \
                                                                              \
   private:                                                                   \
    uint32_t output_idx;                                                      \
  };

DEFINE_SIMPLE_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_AGGREGATOR(Max, max, MAX)

struct DecimalSumOrAvgAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  DecimalSumOrAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType,
      const bool isAvg)
      : Aggregator(
            step,
            cudf::aggregation::SUM,
            inputIndex,
            constant,
            isGlobal,
            resultType),
        isAvg_(isAvg) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    VELOX_FAIL(
        "DecimalSumOrAvgAggregator requires stream-aware addGroupbyRequest");
  }

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    logHashAggDebug(
        "Decimal.addGroupbyRequest.begin",
        step,
        stream,
        tbl.num_rows(),
        tbl.num_columns(),
        inputIndex);
    if (step == core::AggregationNode::Step::kIntermediate &&
        tbl.column(inputIndex).type().id() == cudf::type_id::STRING) {
      auto scale = resultType->isDecimal()
          ? getDecimalPrecisionScale(*resultType).second
          : 0;
      logHashAggDebug(
          "Decimal.addGroupbyRequest.intermediate.deserialize",
          step,
          stream,
          tbl.num_rows(),
          tbl.num_columns(),
          scale);
      auto decoded = cudf_velox::deserializeDecimalSumStateWithCount(
          tbl.column(inputIndex), scale, stream);
      decodedSum_ = std::move(decoded.sum);
      decodedCount_ = std::move(decoded.count);
      if (decodedSum_) {
        logColumnViewDebug(
            "Decimal.addGroupbyRequest.intermediate.sumDecoded",
            step,
            stream,
            decodedSum_->view(),
            requests.size());
      }
      if (decodedCount_) {
        logColumnViewDebug(
            "Decimal.addGroupbyRequest.intermediate.countDecoded",
            step,
            stream,
            decodedCount_->view(),
            requests.size());
      }
      logCudaCheckpoint(
          "Decimal.addGroupbyRequest.intermediate.afterDeserialize",
          step,
          stream,
          tbl.num_rows(),
          tbl.num_columns(),
          scale);

      sumIdx_ = requests.size();
      auto& sumRequest = requests.emplace_back();
      sumRequest.values = decodedSum_->view();
      sumRequest.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());

      countIdx_ = requests.size();
      auto& countRequest = requests.emplace_back();
      countRequest.values = decodedCount_->view();
      countRequest.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      return;
    }

    if (step == core::AggregationNode::Step::kFinal &&
        tbl.column(inputIndex).type().id() == cudf::type_id::STRING) {
      auto scale = getDecimalPrecisionScale(*resultType).second;
      if (isAvg_) {
        logHashAggDebug(
            "Decimal.addGroupbyRequest.finalAvg.deserialize",
            step,
            stream,
            tbl.num_rows(),
            tbl.num_columns(),
            scale);
        auto decoded = cudf_velox::deserializeDecimalSumStateWithCount(
            tbl.column(inputIndex), scale, stream);
        decodedSum_ = std::move(decoded.sum);
        decodedCount_ = std::move(decoded.count);
        if (decodedSum_) {
          logColumnViewDebug(
              "Decimal.addGroupbyRequest.finalAvg.sumDecoded",
              step,
              stream,
              decodedSum_->view(),
              requests.size());
        }
        if (decodedCount_) {
          logColumnViewDebug(
              "Decimal.addGroupbyRequest.finalAvg.countDecoded",
              step,
              stream,
              decodedCount_->view(),
              requests.size());
        }
        logCudaCheckpoint(
            "Decimal.addGroupbyRequest.finalAvg.afterDeserialize",
            step,
            stream,
            tbl.num_rows(),
            tbl.num_columns(),
            scale);

        sumIdx_ = requests.size();
        auto& sumRequest = requests.emplace_back();
        sumRequest.values = decodedSum_->view();
        sumRequest.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        countIdx_ = requests.size();
        auto& countRequest = requests.emplace_back();
        countRequest.values = decodedCount_->view();
        countRequest.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        return;
      } else {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        logHashAggDebug(
            "Decimal.addGroupbyRequest.finalSum.deserialize",
            step,
            stream,
            tbl.num_rows(),
            tbl.num_columns(),
            scale);
        decodedSum_ = cudf_velox::deserializeDecimalSumState(
            tbl.column(inputIndex), scale, stream);
        if (decodedSum_) {
          logColumnViewDebug(
              "Decimal.addGroupbyRequest.finalSum.sumDecoded",
              step,
              stream,
              decodedSum_->view(),
              requests.size());
        }
        logCudaCheckpoint(
            "Decimal.addGroupbyRequest.finalSum.afterDeserialize",
            step,
            stream,
            tbl.num_rows(),
            tbl.num_columns(),
            scale);
        request.values = decodedSum_->view();
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        return;
      }
    } else {
      if (isAvg_ &&
          (step == core::AggregationNode::Step::kPartial ||
           step == core::AggregationNode::Step::kSingle)) {
        // Isolate decimal AVG sum/count into independent requests. This avoids
        // relying on multi-aggregation request behavior for decimal inputs.
        sumIdx_ = requests.size();
        auto& sumRequest = requests.emplace_back();
        sumRequest.values = tbl.column(inputIndex);
        sumRequest.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        countIdx_ = requests.size();
        auto& countRequest = requests.emplace_back();
        countRequest.values = tbl.column(inputIndex);
        countRequest.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
      } else {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        if (step == core::AggregationNode::Step::kPartial) {
          request.aggregations.push_back(
              cudf::make_count_aggregation<cudf::groupby_aggregation>(
                  cudf::null_policy::EXCLUDE));
        }
      }
      logHashAggDebug(
          "Decimal.addGroupbyRequest.rawInput",
          step,
          stream,
          tbl.num_rows(),
          tbl.num_columns(),
          requests.size());
      return;
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    logHashAggDebug(
        "Decimal.makeOutputColumn.begin",
        step,
        stream,
        results.size(),
        isAvg_ ? 1 : 0,
        sumIdx_);
    auto col = std::move(results[sumIdx_].results[0]);
    auto maybeValidateStateRoundtrip =
        [&](cudf::column_view const& sumView,
            cudf::column_view const& countView,
            cudf::column_view const& stateView) {
          if (!hashAggDebugStateRoundtripValidate()) {
            return;
          }
          auto const& cfg = facebook::velox::cudf_velox::CudfConfig::getInstance();
          int32_t scale = static_cast<int32_t>(sumView.type().scale());
          if (scale < 0) {
            scale = -scale;
          }
          auto roundtrip = validateDecimalStateRoundtrip(
              sumView,
              countView,
              stateView,
              scale,
              stream,
              cfg.debugHashAggEndToEndMaxRows,
              cfg.debugHashAggEndToEndBatchRows);
          if (roundtrip.skipped) {
            LOG(INFO) << "[HashAggStateRoundtrip] skipped reason="
                      << roundtrip.reason << " step=" << stepName(step)
                      << " rows=" << sumView.size();
            return;
          }
          LOG(INFO) << "[HashAggStateRoundtrip] step=" << stepName(step)
                    << " rows=" << roundtrip.rows
                    << " checked=" << roundtrip.checked
                    << " maskMismatches=" << roundtrip.maskMismatches
                    << " sumMismatches=" << roundtrip.sumMismatches
                    << " countMismatches=" << roundtrip.countMismatches;
          if (roundtrip.maskMismatches > 0 || roundtrip.sumMismatches > 0 ||
              roundtrip.countMismatches > 0) {
            LOG(INFO) << "[HashAggStateRoundtrip] firstMismatch row="
                      << roundtrip.firstRow
                      << " expectedValid=" << roundtrip.expectedValid
                      << " actualValid=" << roundtrip.actualValid
                      << " expectedSum=" << toString128(roundtrip.expectedSum)
                      << " actualSum=" << toString128(roundtrip.actualSum)
                      << " expectedCount=" << roundtrip.expectedCount
                      << " actualCount=" << roundtrip.actualCount;
          }
        };
    if (isAvg_ && step == core::AggregationNode::Step::kSingle) {
      auto count = std::move(results[countIdx_].results[0]);
      return computeAvgColumn(std::move(col), std::move(count), stream);
    }
    if (step == core::AggregationNode::Step::kPartial) {
      std::unique_ptr<cudf::column> count;
      if (isAvg_) {
        count = std::move(results[countIdx_].results[0]);
      } else {
        count = std::move(results[sumIdx_].results[1]);
      }
      if (count->type().id() != cudf::type_id::INT64) {
        count = cudf::cast(*count, cudf::data_type{cudf::type_id::INT64}, stream);
      }
      auto stateCol = cudf_velox::serializeDecimalSumState(
          col->view(), count->view(), stream);
      maybeValidateStateRoundtrip(col->view(), count->view(), stateCol->view());
      return stateCol;
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      if (count->type().id() != cudf::type_id::INT64) {
        count = cudf::cast(*count, cudf::data_type{cudf::type_id::INT64}, stream);
      }
      auto stateCol = cudf_velox::serializeDecimalSumState(
          col->view(), count->view(), stream);
      maybeValidateStateRoundtrip(col->view(), count->view(), stateCol->view());
      return stateCol;
    }
    if (isAvg_ && step == core::AggregationNode::Step::kFinal) {
      auto count = std::move(results[countIdx_].results[0]);
      return computeAvgColumn(std::move(col), std::move(count), stream);
    }
    auto const cudfResType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfResType) {
      col = cudf::cast(*col, cudfResType, stream);
    }
    return col;
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
    logHashAggDebug(
        "Decimal.doReduce.begin",
        step,
        stream,
        input.num_rows(),
        input.num_columns(),
        inputIndex);
    logCudaCheckpoint(
        "Decimal.doReduce.begin.cudaCheckpoint",
        step,
        stream,
        input.num_rows(),
        input.num_columns(),
        inputIndex);
    if (step == core::AggregationNode::Step::kSingle && isAvg_) {
      auto const sumAgg =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      cudf::column_view inputCol = input.column(inputIndex);
      auto sumScalar = cudf::reduce(inputCol, *sumAgg, inputCol.type(), stream);
      auto countAgg = cudf::make_count_aggregation<cudf::reduce_aggregation>(
          cudf::null_policy::EXCLUDE);
      auto countScalar = cudf::reduce(
          inputCol,
          *countAgg,
          cudf::data_type{cudf::type_id::INT64},
          stream);
      auto sumCol = cudf::make_column_from_scalar(*sumScalar, 1, stream);
      auto countCol = cudf::make_column_from_scalar(*countScalar, 1, stream);
      logHashAggDebug("Decimal.doReduce.singleAvg.compute", step, stream, 1, 2, 0);
      return computeAvgColumn(std::move(sumCol), std::move(countCol), stream);
    }
    auto const aggRequest =
        cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    cudf::column_view inputCol = input.column(inputIndex);
    logColumnViewDebug("Decimal.doReduce.inputColumn", step, stream, inputCol, inputIndex);
    if (step == core::AggregationNode::Step::kPartial) {
      auto sumScalar =
          cudf::reduce(inputCol, *aggRequest, inputCol.type(), stream);
      auto countAgg = cudf::make_count_aggregation<cudf::reduce_aggregation>(
          cudf::null_policy::EXCLUDE);
      auto countScalar = cudf::reduce(
          inputCol,
          *countAgg,
          cudf::data_type{cudf::type_id::INT64},
          stream);
      auto sumCol = cudf::make_column_from_scalar(*sumScalar, 1, stream);
      auto countCol = cudf::make_column_from_scalar(*countScalar, 1, stream);
      logHashAggDebug(
          "Decimal.doReduce.partial.serialize",
          step,
          stream,
          1,
          2,
          0);
      return cudf_velox::serializeDecimalSumState(
          sumCol->view(), countCol->view(), stream);
    }
    if (step == core::AggregationNode::Step::kIntermediate &&
        inputCol.type().id() == cudf::type_id::STRING) {
      auto scale = outputType->isDecimal()
          ? getDecimalPrecisionScale(*outputType).second
          : 0;
      auto decoded =
          cudf_velox::deserializeDecimalSumStateWithCount(inputCol, scale, stream);
      if (decoded.sum) {
        logColumnViewDebug(
            "Decimal.doReduce.intermediate.sumDecoded",
            step,
            stream,
            decoded.sum->view(),
            inputIndex);
      }
      if (decoded.count) {
        logColumnViewDebug(
            "Decimal.doReduce.intermediate.countDecoded",
            step,
            stream,
            decoded.count->view(),
            inputIndex);
      }
      logCudaCheckpoint(
          "Decimal.doReduce.intermediate.afterDeserialize",
          step,
          stream,
          decoded.sum ? decoded.sum->size() : 0,
          decoded.count ? decoded.count->size() : 0,
          scale);
      logHashAggDebug(
          "Decimal.doReduce.intermediate.deserialize",
          step,
          stream,
          decoded.sum ? decoded.sum->size() : 0,
          decoded.count ? decoded.count->size() : 0,
          scale);
      auto sumScalar = cudf::reduce(
          decoded.sum->view(),
          *aggRequest,
          decoded.sum->view().type(),
          stream);
      auto countScalar = cudf::reduce(
          decoded.count->view(),
          *aggRequest,
          cudf::data_type{cudf::type_id::INT64},
          stream);
      auto sumCol = cudf::make_column_from_scalar(*sumScalar, 1, stream);
      auto countCol = cudf::make_column_from_scalar(*countScalar, 1, stream);
      logHashAggDebug(
          "Decimal.doReduce.intermediate.serialize",
          step,
          stream,
          1,
          2,
          scale);
      return cudf_velox::serializeDecimalSumState(
          sumCol->view(), countCol->view(), stream);
    }
    if (step == core::AggregationNode::Step::kFinal &&
        inputCol.type().id() == cudf::type_id::STRING) {
      auto scale = getDecimalPrecisionScale(*outputType).second;
      if (isAvg_) {
        // AVG
        // deserialize the results (sum and count)
        auto sumAndCount = cudf_velox::deserializeDecimalSumStateWithCount(inputCol, scale, stream);
        if (sumAndCount.sum) {
          logColumnViewDebug(
              "Decimal.doReduce.finalAvg.sumDecoded",
              step,
              stream,
              sumAndCount.sum->view(),
              inputIndex);
        }
        if (sumAndCount.count) {
          logColumnViewDebug(
              "Decimal.doReduce.finalAvg.countDecoded",
              step,
              stream,
              sumAndCount.count->view(),
              inputIndex);
        }
        logCudaCheckpoint(
            "Decimal.doReduce.finalAvg.afterDeserialize",
            step,
            stream,
            sumAndCount.sum ? sumAndCount.sum->size() : 0,
            sumAndCount.count ? sumAndCount.count->size() : 0,
            scale);
        logHashAggDebug(
            "Decimal.doReduce.finalAvg.deserialize",
            step,
            stream,
            sumAndCount.sum ? sumAndCount.sum->size() : 0,
            sumAndCount.count ? sumAndCount.count->size() : 0,
            scale);
        // reduce the two results to get final sum and count scalars
        auto sumScalar = cudf::reduce(sumAndCount.sum->view(), *aggRequest, sumAndCount.sum->view().type(), stream);
        auto countScalar = cudf::reduce(sumAndCount.count->view(), *aggRequest, cudf::data_type{cudf::type_id::INT64}, stream);
        // convert to columns in order to perform division, as we cannot divide scalars directly
        auto sumCol = cudf::make_column_from_scalar(*sumScalar, 1, stream);
        auto countCol = cudf::make_column_from_scalar(*countScalar, 1, stream);
        logHashAggDebug(
            "Decimal.doReduce.finalAvg.compute",
            step,
            stream,
            1,
            2,
            scale);
        return computeAvgColumn(std::move(sumCol), std::move(countCol), stream);
      } else {
        // SUM
        decodedSum_ = cudf_velox::deserializeDecimalSumState(
            inputCol, scale, stream);
        if (decodedSum_) {
          logColumnViewDebug(
              "Decimal.doReduce.finalSum.sumDecoded",
              step,
              stream,
              decodedSum_->view(),
              inputIndex);
        }
        logCudaCheckpoint(
            "Decimal.doReduce.finalSum.afterDeserialize",
            step,
            stream,
            decodedSum_ ? decodedSum_->size() : 0,
            1,
            scale);
        logHashAggDebug(
            "Decimal.doReduce.finalSum.deserialize",
            step,
            stream,
            decodedSum_ ? decodedSum_->size() : 0,
            1,
            scale);
        inputCol = decodedSum_->view();
        // @TODO does this need to drop through to the code below
        // or can we just do that stuff here, and not need decodedSum_ or decodedCount_
        // why we do have those anyway if they're only set in addGroupbyRequest() and
        // either overwritten or not even used here?
        // what does the final cudf::reduce() below actually do?
      }
    }
    auto const cudfOutType = cudf_velox::veloxToCudfDataType(outputType);
    std::unique_ptr<cudf::column> castedInput;
    if (outputType->isDecimal() && inputCol.type() != cudfOutType) {
      castedInput = cudf::cast(inputCol, cudfOutType, stream);
      inputCol = castedInput->view();
    }
    auto const resultScalar =
        cudf::reduce(inputCol, *aggRequest, cudfOutType, stream);
    return cudf::make_column_from_scalar(*resultScalar, 1, stream);
  }

 private:
  std::unique_ptr<cudf::column> computeAvgColumn(
      std::unique_ptr<cudf::column> sum,
      std::unique_ptr<cudf::column> count,
      rmm::cuda_stream_view stream) const {
    if (hashAggDebugEnabled()) {
      LOG(INFO) << "[CudfHashAggDebug] stage=Decimal.computeAvgColumn.begin step="
                << stepName(step) << " stream="
                << reinterpret_cast<const void*>(stream.value()) << " sumType="
                << static_cast<int>(sum->type().id()) << " countType="
                << static_cast<int>(count->type().id()) << " sumRows="
                << sum->size() << " countRows=" << count->size();
      logColumnViewDebug(
          "Decimal.computeAvgColumn.sumInput", step, stream, sum->view(), 0);
      logColumnViewDebug(
          "Decimal.computeAvgColumn.countInput", step, stream, count->view(), 1);
    }
    logCudaCheckpoint(
        "Decimal.computeAvgColumn.beforeCountCast",
        step,
        stream,
        sum->size(),
        count->size(),
        0);
    if (count->type().id() != cudf::type_id::INT64) {
      count = cudf::cast(*count, cudf::data_type{cudf::type_id::INT64}, stream);
      logColumnViewDebug(
          "Decimal.computeAvgColumn.countCastOutput",
          step,
          stream,
          count->view(),
          1);
    }
    logCudaCheckpoint(
        "Decimal.computeAvgColumn.beforeComputeDecimalAverage",
        step,
        stream,
        sum->size(),
        count->size(),
        0);
    auto avgCol = cudf_velox::computeDecimalAverage(
        sum->view(), count->view(), stream);
    logCudaCheckpoint(
        "Decimal.computeAvgColumn.afterComputeDecimalAverage",
        step,
        stream,
        avgCol ? avgCol->size() : 0,
        1,
        0);
    auto const cudfOutType = cudf_velox::veloxToCudfDataType(resultType);
    if (avgCol->type() != cudfOutType) {
      logCudaCheckpoint(
          "Decimal.computeAvgColumn.beforeOutputCast",
          step,
          stream,
          avgCol->size(),
          1,
          static_cast<int>(avgCol->type().id()));
      avgCol = cudf::cast(avgCol->view(), cudfOutType, stream);
      logCudaCheckpoint(
          "Decimal.computeAvgColumn.afterOutputCast",
          step,
          stream,
          avgCol->size(),
          1,
          static_cast<int>(avgCol->type().id()));
    }
    if (hashAggDebugEnabled()) {
      LOG(INFO) << "[CudfHashAggDebug] stage=Decimal.computeAvgColumn.end step="
                << stepName(step) << " stream="
                << reinterpret_cast<const void*>(stream.value()) << " outType="
                << static_cast<int>(avgCol->type().id()) << " outRows="
                << avgCol->size();
      logColumnViewDebug(
          "Decimal.computeAvgColumn.output", step, stream, avgCol->view(), 2);
    }
    return avgCol;
  }

  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  const bool isAvg_{false};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
};

struct CountAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  CountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType)
      : Aggregator(
            step,
            cudf::aggregation::COUNT_VALID,
            inputIndex,
            constant,
            isGlobal,
            resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    auto& request = requests.emplace_back();
    outputIdx_ = requests.size() - 1;
    request.values = tbl.column(constant == nullptr ? inputIndex : 0);
    std::unique_ptr<cudf::groupby_aggregation> aggRequest =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              constant == nullptr ? cudf::null_policy::EXCLUDE
                                  : cudf::null_policy::INCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    request.aggregations.push_back(std::move(aggRequest));
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
    if (exec::isRawInput(step)) {
      // For raw input, implement count using size + null count
      auto inputCol = input.column(constant == nullptr ? inputIndex : 0);

      // count_valid: size - null_count, count_all: just the size
      int64_t count = constant == nullptr
          ? inputCol.size() - inputCol.null_count()
          : inputCol.size();

      auto resultScalar = cudf::numeric_scalar<int64_t>(count);

      return cudf::make_column_from_scalar(resultScalar, 1, stream);
    } else {
      // For non-raw input (intermediate/final), use sum aggregation
      auto const aggRequest =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto const cudfOutputType = cudf::data_type(cudf::type_id::INT64);
      auto const resultScalar = cudf::reduce(
          input.column(inputIndex), *aggRequest, cudfOutputType, stream);
      resultScalar->set_valid_async(true, stream);
      return cudf::make_column_from_scalar(*resultScalar, 1, stream);
    }
    return nullptr;
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    // cudf produces int32 for count(0) but velox expects int64
    auto col = std::move(results[outputIdx_].results[0]);
    const auto cudfOutputType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream);
    }
    return col;
  }

 private:
  uint32_t outputIdx_;
};

struct MeanAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  MeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& resultType)
      : Aggregator(
            step,
            cudf::aggregation::MEAN,
            inputIndex,
            constant,
            isGlobal,
            resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto& request = requests.emplace_back();
        meanIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        break;
      }
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal: {
        // In intermediate and final aggregation, the previously computed sum
        // and count are in the child columns of the input column.
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex).child(0);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto& request2 = requests.emplace_back();
        countIdx_ = requests.size() - 1;
        request2.values = tbl.column(inputIndex).child(1);
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        request2.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        // We don't know how to handle kIntermediate step for mean
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[meanIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[sumIdx_].results[1]);

        auto const size = sum->size();
        auto const cudfSumType = cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType = cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudfSumType) {
          sum = cudf::cast(*sum, cudfSumType, stream);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(*count, cudf::data_type(cudfCountType), stream);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kIntermediate: {
        // The difference between intermediate and partial is in where the
        // sum and count are coming from. In partial, since the input column is
        // the same, the sum and count are in the same agg result. In
        // intermediate, the input columns are different (it's the child
        // columns of the input column) and so the sum and count are in
        // different agg results.
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);

        auto size = sum->size();
        auto const cudfSumType = cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType = cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudfSumType) {
          sum = cudf::cast(*sum, cudfSumType, stream);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count = cudf::cast(*count, cudf::data_type(cudfCountType), stream);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            cudf_velox::veloxToCudfDataType(resultType),
            stream);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto const aggRequest =
            cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType);
        auto const resultScalar = cudf::reduce(
            input.column(inputIndex), *aggRequest, cudfOutputType, stream);
        return cudf::make_column_from_scalar(*resultScalar, 1, stream);
      }
      case core::AggregationNode::Step::kPartial: {
        VELOX_CHECK(outputType->isRow());
        auto const& rowType = outputType->asRow();
        auto const sumType = rowType.childAt(0);
        auto const countType = rowType.childAt(1);
        auto const cudfSumType = cudf_velox::veloxToCudfDataType(sumType);
        auto const cudfCountType = cudf_velox::veloxToCudfDataType(countType);

        // sum
        auto const aggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar = cudf::reduce(
            input.column(inputIndex), *aggRequest, cudfSumType, stream);
        auto sumCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream);

        // libcudf doesn't have a count agg for reduce. What we want is to
        // count the number of valid rows.
        auto countCol = cudf::make_column_from_scalar(
            cudf::numeric_scalar<int64_t>(
                input.column(inputIndex).size() -
                input.column(inputIndex).null_count()),
            1,
            stream);

        // Assemble into struct as expected by velox.
        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sumCol));
        children.push_back(std::move(countCol));
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            1,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        // Input column has two children: sum and count
        auto const sumCol = input.column(inputIndex).child(0);
        auto const countCol = input.column(inputIndex).child(1);

        // sum the sums
        auto const sumAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar =
            cudf::reduce(sumCol, *sumAggRequest, sumCol.type(), stream);
        auto sumResultCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream);

        // sum the counts
        auto const countAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const countResultScalar =
            cudf::reduce(countCol, *countAggRequest, countCol.type(), stream);

        // divide the sums by the counts
        auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType);
        return cudf::binary_operation(
            *sumResultCol,
            *countResultScalar,
            cudf::binary_operator::DIV,
            cudfOutputType,
            stream);
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  uint32_t meanIdx_;
  uint32_t sumIdx_;
  uint32_t countIdx_;
};

std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator> createAggregator(
    core::AggregationNode::Step step,
    std::string const& kind,
    uint32_t inputIndex,
    VectorPtr constant,
    bool isGlobal,
    const TypePtr& resultType,
    const std::vector<TypePtr>& rawInputTypes = {}) {
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    bool isDecimalInput =
        rawInputTypes.size() == 1 && rawInputTypes[0]->isDecimal();
    if (isDecimalInput) {
      return std::make_unique<DecimalSumOrAvgAggregator>(
          step, inputIndex, constant, isGlobal, resultType, false);
    }
    return std::make_unique<SumAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    return std::make_unique<CountAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<MinAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<MaxAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    bool isDecimalInput =
        rawInputTypes.size() == 1 && rawInputTypes[0]->isDecimal();
    if (isDecimalInput) {
      return std::make_unique<DecimalSumOrAvgAggregator>(
          step, inputIndex, constant, isGlobal, resultType, true);
    }
    return std::make_unique<MeanAggregator>(
        step, inputIndex, constant, isGlobal, resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported");
  }
}

/// \brief Convert companion function to step for the aggregation function
///
/// Companion functions are functions that are registered in velox along with
/// their main aggregation functions. These are designed to always function
/// with a fixed `step`. This is to allow spark style planNodes where `step` is
/// the property of the aggregation function rather than the planNode.
/// Companion functions allow us to override the planNode's step and use
/// aggregations of different steps in the same planNode
/// If an agg function name contains companionStep keyword, may cause error, now
/// it does not exist.
core::AggregationNode::Step getCompanionStep(
    std::string const& kind,
    core::AggregationNode::Step step) {
  if (kind.ends_with("_merge")) {
    return core::AggregationNode::Step::kIntermediate;
  }

  if (kind.ends_with("_partial")) {
    return core::AggregationNode::Step::kPartial;
  }

  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (kind.find("_merge_extract") != std::string::npos) {
    return core::AggregationNode::Step::kFinal;
  }

  return step;
}

std::string getOriginalName(const std::string& kind) {
  if (kind.ends_with("_merge")) {
    return kind.substr(0, kind.size() - std::string("_merge").size());
  }

  if (kind.ends_with("_partial")) {
    return kind.substr(0, kind.size() - std::string("_partial").size());
  }
  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (auto pos = kind.find("_merge_extract"); pos != std::string::npos) {
    return kind.substr(0, pos);
  }

  return kind;
}

bool hasFinalAggs(
    std::vector<core::AggregationNode::Aggregate> const& aggregates) {
  return std::any_of(aggregates.begin(), aggregates.end(), [](auto const& agg) {
    return agg.call->name().ends_with("_merge_extract");
  });
}

auto toAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto const step = aggregationNode.step();
  bool const isGlobal = aggregationNode.groupingKeys().empty();
  auto const& inputRowSchema = aggregationNode.sources()[0]->outputType();
  const auto numKeys = aggregationNode.groupingKeys().size();
  const auto outputType = aggregationNode.outputType();

  std::vector<std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator>>
      aggregators;
  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    std::vector<column_index_t> aggInputs;
    std::vector<VectorPtr> aggConstants;
    for (auto const& arg : aggregate.call->inputs()) {
      if (auto const field =
              dynamic_cast<core::FieldAccessTypedExpr const*>(arg.get())) {
        aggInputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else if (
          auto constant =
              dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
        aggInputs.push_back(kConstantChannel);
        aggConstants.push_back(constant->toConstantVector(operatorCtx.pool()));
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }
    // The loop on aggregate.call->inputs() is taken from
    // AggregateInfo.cpp::toAggregateInfo(). It seems to suggest that there can
    // be multiple inputs to an aggregate.
    // We're postponing properly supporting this for now because the currently
    // supported aggregation functions in cudf_velox don't use it.
    VELOX_CHECK(aggInputs.size() <= 1);
    if (aggInputs.empty()) {
      aggInputs.push_back(0);
    }

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    auto const kind = aggregate.call->name();
    auto const inputIndex = aggInputs[0];
    auto const constant = aggConstants.empty() ? nullptr : aggConstants[0];
    auto const companionStep = getCompanionStep(kind, step);
    const auto originalName = getOriginalName(kind);
    const auto resultType = exec::isPartialOutput(companionStep)
        ? exec::resolveIntermediateType(originalName, aggregate.rawInputTypes)
        : outputType->childAt(numKeys + i);

    aggregators.push_back(createAggregator(
        companionStep,
        kind,
        inputIndex,
        constant,
        isGlobal,
        resultType,
        aggregate.rawInputTypes));
  }
  return aggregators;
}

auto toIntermediateAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto const step = core::AggregationNode::Step::kIntermediate;
  bool const isGlobal = aggregationNode.groupingKeys().empty();
  auto const& inputRowSchema = aggregationNode.outputType();

  std::vector<std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator>>
      aggregators;
  for (size_t i = 0; i < aggregationNode.aggregates().size(); i++) {
    // Intermediate aggregation has a 1:1 mapping between input and output.
    // We don't need to figure out input from the aggregate function.
    auto const& aggregate = aggregationNode.aggregates()[i];
    auto const inputIndex = aggregationNode.groupingKeys().size() + i;
    auto const kind = aggregate.call->name();
    auto const constant = nullptr;
    const auto originalName = getOriginalName(kind);
    auto const companionStep = getCompanionStep(kind, step);
    if (exec::isPartialOutput(companionStep)) {
      const auto resultType =
          exec::resolveIntermediateType(originalName, aggregate.rawInputTypes);
    aggregators.push_back(createAggregator(
        step,
        kind,
        inputIndex,
        constant,
        isGlobal,
        resultType,
        aggregate.rawInputTypes));
    } else {
      // Final step aggregator will not use the intermediate aggregator.
      aggregators.push_back(nullptr);
    }
  }
  return aggregators;
}

} // namespace

namespace facebook::velox::cudf_velox {

CudfHashAggregation::CudfHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          aggregationNode->step() == core::AggregationNode::Step::kPartial
              ? "CudfPartialAggregation"
              : "CudfAggregation",
          aggregationNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt),
      NvtxHelper(
          nvtx3::rgb{34, 139, 34}, // Forest Green
          operatorId,
          fmt::format("[{}]", aggregationNode->id())),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isGlobal_(aggregationNode->groupingKeys().empty()),
      isDistinct_(!isGlobal_ && aggregationNode->aggregates().empty()),
      step_(aggregationNode->step()),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfHashAggregation::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      groupingKeyInputChannels_, groupingKeyOutputChannels_);

  auto const numGroupingKeys = groupingKeyOutputChannels_.size();

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  aggregators_ = toAggregators(*aggregationNode_, *operatorCtx_);
  intermediateAggregators_ =
      toIntermediateAggregators(*aggregationNode_, *operatorCtx_);

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfHashAggregation::setupGroupingKeyChannelProjections(
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) const {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  auto const& inputType = aggregationNode_->sources()[0]->outputType();
  auto const& groupingKeys = aggregationNode_->groupingKeys();
  // The map from the grouping key output channel to the input channel.
  //
  // NOTE: grouping key output order is specified as 'groupingKeys' in
  // 'aggregationNode_'.
  std::vector<exec::IdentityProjection> groupingKeyProjections;
  groupingKeyProjections.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyProjections.emplace_back(
        exec::exprToChannel(groupingKeys[i].get(), inputType), i);
  }

  groupingKeyInputChannels.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyInputChannels.push_back(groupingKeyProjections[i].inputChannel);
  }

  groupingKeyOutputChannels.resize(groupingKeys.size());

  std::iota(
      groupingKeyOutputChannels.begin(), groupingKeyOutputChannels.end(), 0);
}

void CudfHashAggregation::computeIntermediateGroupbyPartial(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto groupbyOnInput = doGroupByAggregation(
      tbl->getTableView(),
      groupingKeyInputChannels_,
      aggregators_,
      inputTableStream);

  // If we already have partial output, concatenate the new results with it.
  if (partialOutput_) {
    // Create a vector of tables to concatenate
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(partialOutput_->getTableView());
    tablesToConcat.push_back(groupbyOnInput->getTableView());

    auto partialOutputStream = partialOutput_->stream();
    // We need to join the input table stream on the partial output stream to
    // make sure the intermediate results are available when we do the concat.
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    // Concatenate the tables
    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream);

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        partialOutputStream);
    partialOutput_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    partialOutput_ = groupbyOnInput;
  }
}

void CudfHashAggregation::computeIntermediateDistinctPartial(
    CudfVectorPtr tbl) {
  // For every input, we'll concat with existing distinct results and then do a
  // distinct on the concatenated results.

  auto inputTableStream = tbl->stream();

  if (partialOutput_) {
    // Concatenate the input table with the existing distinct results.
    std::vector<cudf::table_view> tablesToConcat;
    tablesToConcat.push_back(partialOutput_->getTableView());
    tablesToConcat.push_back(tbl->getTableView().select(
        groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end()));

    auto partialOutputStream = partialOutput_->stream();
    // We need to join the input table stream on the partial output stream to
    // make sure the input table is available when we do the concat.
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputTableStream},
        partialOutputStream);

    auto concatenatedTable =
        cudf::concatenate(tablesToConcat, partialOutputStream);

    // Do a distinct on the concatenated results.
    // Keep concatenatedTable alive while we use its view.
    auto distinctOutput = getDistinctKeys(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        inputTableStream);
    partialOutput_ = distinctOutput;
  } else {
    // First time processing, just store the result of the input batch's
    // distinct. Use getTableView() to avoid expensive materialization for
    // packed_table. tbl stays alive during this function call.
    partialOutput_ = getDistinctKeys(
        tbl->getTableView(), groupingKeyInputChannels_, inputTableStream);
  }
}

void CudfHashAggregation::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() == 0) {
    return;
  }
  if (hashAggDebugEnabled()) {
    LOG(INFO) << "[CudfHashAggDebug] stage=addInput.begin step="
              << stepName(step_) << " rows=" << input->size()
              << " isPartialOutput=" << isPartialOutput_
              << " isGlobal=" << isGlobal_ << " isDistinct=" << isDistinct_;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  maybeDeviceSyncProbe(
      kDeviceSyncGroupOperatorBoundary,
      2,
      "HashAgg.addInput.begin",
      step_,
      cudfInput->stream(),
      input->size(),
      input->type()->size(),
      numInputRows_);

  if (isPartialOutput_ && !isGlobal_) {
    if (isDistinct_) {
      // Handle partial distinct aggregation.
      computeIntermediateDistinctPartial(cudfInput);
    } else {
      // Handle partial groupby aggregation.
      computeIntermediateGroupbyPartial(cudfInput);
    }
    if (hashAggDebugEnabled()) {
      LOG(INFO) << "[CudfHashAggDebug] stage=addInput.partialProcessed step="
                << stepName(step_)
                << " partialOutputRows="
                << (partialOutput_ ? partialOutput_->size() : 0)
                << " numInputRows=" << numInputRows_;
    }
    maybeDeviceSyncProbe(
        kDeviceSyncGroupOperatorBoundary,
        3,
        "HashAgg.addInput.partialProcessed",
        step_,
        cudfInput->stream(),
        partialOutput_ ? partialOutput_->size() : 0,
        isDistinct_ ? 1 : 0,
        numInputRows_);
    return;
  }

  // Handle final aggregation or global cases.
  inputs_.push_back(std::move(cudfInput));
  if (hashAggDebugEnabled()) {
    LOG(INFO) << "[CudfHashAggDebug] stage=addInput.buffered step="
              << stepName(step_)
              << " bufferedInputs=" << inputs_.size() << " numInputRows="
              << numInputRows_;
  }
  maybeDeviceSyncProbe(
      kDeviceSyncGroupOperatorBoundary,
      4,
      "HashAgg.addInput.buffered",
      step_,
      inputs_.back()->stream(),
      inputs_.back()->size(),
      inputs_.size(),
      numInputRows_);
}

CudfVectorPtr CudfHashAggregation::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<Aggregator>>& aggregators,
    rmm::cuda_stream_view stream) {
  logHashAggDebug(
      "HashAgg.doGroupByAggregation.begin",
      step_,
      stream,
      tableView.num_rows(),
      tableView.num_columns(),
      groupByKeys.size());
  logCudaCheckpoint(
      "HashAgg.doGroupByAggregation.begin.cudaCheckpoint",
      step_,
      stream,
      tableView.num_rows(),
      tableView.num_columns(),
      groupByKeys.size());
  logTableViewDebug("HashAgg.doGroupByAggregation.inputTable", step_, stream, tableView);
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());

  size_t const numGroupingKeys = groupbyKeyView.num_columns();

  auto const nullPolicy =
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby groupByOwner(groupbyKeyView, nullPolicy);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (size_t aggregatorIdx = 0; aggregatorIdx < aggregators.size();
       ++aggregatorIdx) {
    auto& aggregator = aggregators[aggregatorIdx];
    VELOX_CHECK_NOT_NULL(aggregator);
    auto const requestsBefore = requests.size();
    if (hashAggDebugEnabled()) {
      LOG(INFO)
          << "[CudfHashAggDebug] stage=HashAgg.addGroupbyRequest.dispatch step="
          << stepName(step_) << " stream="
          << reinterpret_cast<const void*>(stream.value()) << " aggregatorIdx="
          << aggregatorIdx
          << " aggregatorKind=" << static_cast<int>(aggregator->kind)
          << " aggregatorInputIndex=" << aggregator->inputIndex
          << " requestsBefore=" << requestsBefore;
    }
    aggregator->addGroupbyRequest(tableView, requests, stream);
    if (hashAggDebugEnabled()) {
      LOG(INFO)
          << "[CudfHashAggDebug] stage=HashAgg.addGroupbyRequest.added step="
          << stepName(step_) << " stream="
          << reinterpret_cast<const void*>(stream.value()) << " aggregatorIdx="
          << aggregatorIdx << " requestsAfter=" << requests.size()
          << " requestDelta=" << (requests.size() - requestsBefore);
    }
  }
  logGroupbyRequestsDebug(step_, stream, tableView, requests);
  auto const decimalCpuDetourEnabled = hashAggDebugDecimalCpuAggregateMode() != 0;
  auto const useDecimalCpuDetour =
      decimalCpuDetourEnabled &&
      (hasDecimalGroupbyRequest(requests) || hasDecimalAggregator(aggregators));
  auto const fakeGroupbyMode = hashAggDebugFakeGroupbyMode();
  auto const fakeModeSupportedStep = (step_ == core::AggregationNode::Step::kFinal ||
                                      step_ == core::AggregationNode::Step::kSingle);
  if (fakeGroupbyMode != 0 && !fakeModeSupportedStep) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "fakeAggregate.skipUnsupportedStep step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << requests.size() << " groupingKeys="
                 << numGroupingKeys << " fakeMode=" << fakeGroupbyMode << "("
                 << fakeGroupbyModeName(fakeGroupbyMode) << ")";
  }
  if (fakeGroupbyMode != 0 && fakeModeSupportedStep) {
    auto const fakeNumRows = tableView.num_rows() > 0 ? 1 : 0;
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "fakeAggregate.begin step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << requests.size() << " groupingKeys="
                 << numGroupingKeys << " fakeMode=" << fakeGroupbyMode << "("
                 << fakeGroupbyModeName(fakeGroupbyMode) << ")"
                 << " fakeRows=" << fakeNumRows;
    if (fakeNumRows == 0) {
      return nullptr;
    }
    std::vector<std::unique_ptr<cudf::column>> fakeColumns;
    fakeColumns.reserve(outputType_->size());
    for (int i = 0; i < outputType_->size(); ++i) {
      auto const cudfOutputType =
          cudf_velox::veloxToCudfDataType(outputType_->childAt(i));
      auto fakeScalar =
          makeFakeConstantScalar(cudfOutputType, fakeGroupbyMode, stream);
      auto fakeColumn =
          cudf::make_column_from_scalar(*fakeScalar, fakeNumRows, stream);
      if (fakeColumn) {
        logColumnViewDebug(
            "HashAgg.doGroupByAggregation.fakeAggregate.outputColumn",
            step_,
            stream,
            fakeColumn->view(),
            i);
      }
      fakeColumns.push_back(std::move(fakeColumn));
    }
    auto fakeTable = std::make_unique<cudf::table>(std::move(fakeColumns));
    logCudaCheckpoint(
        "HashAgg.doGroupByAggregation.fakeAggregate.cudaCheckpoint",
        step_,
        stream,
        fakeNumRows,
        fakeTable->num_columns(),
        fakeGroupbyMode);
    logHashAggDebug(
        "HashAgg.doGroupByAggregation.fakeAggregate.end",
        step_,
        stream,
        fakeNumRows,
        fakeTable->num_columns(),
        fakeGroupbyMode);
    return std::make_shared<cudf_velox::CudfVector>(
        pool(), outputType_, fakeNumRows, std::move(fakeTable), stream);
  }
  maybeDeviceSyncProbe(
      kDeviceSyncGroupGroupByCore,
      1,
      "HashAgg.doGroupByAggregation.preAggregate",
      step_,
      stream,
      tableView.num_rows(),
      requests.size(),
      numGroupingKeys);
  logHashAggDebug(
      "HashAgg.doGroupByAggregation.beforeAggregate",
      step_,
      stream,
      tableView.num_rows(),
      requests.size(),
      numGroupingKeys);
  std::string dumpDir;
  if (!hashAggDebugDumpDir().empty() && hasDecimalGroupbyRequest(requests)) {
    dumpDir = maybeWriteGroupbyDump(
        step_,
        stream,
        groupbyKeyView,
        nullPolicy,
        requests,
        tableView.num_rows(),
        tableView.num_columns(),
        numGroupingKeys);
  }
  logCudaCheckpoint(
      "HashAgg.doGroupByAggregation.beforeAggregate.cudaCheckpoint",
      step_,
      stream,
      tableView.num_rows(),
      requests.size(),
      numGroupingKeys);
  if (!useDecimalCpuDetour) {
    auto const probeFailure = runGroupByAggregateRequestIsolationProbes(
        step_,
        stream,
        tableView,
        groupbyKeyView,
        nullPolicy,
        requests,
        numGroupingKeys);
    if (!probeFailure.empty()) {
      LOG(ERROR) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "aggregateProbe.firstFailure step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << requests.size() << " groupingKeys="
                 << numGroupingKeys << " detail=" << probeFailure;
      VELOX_FAIL(
          "HashAgg aggregate request-isolation probe failed before main "
          "aggregate: {}",
          probeFailure);
    }
  }
  std::unique_ptr<cudf::table> groupKeys;
  std::vector<cudf::groupby::aggregation_result> results;
  if (useDecimalCpuDetour) {
    std::string detourError;
    if (!runDecimalCpuAggregateDetour(
            pool(),
            step_,
            stream,
            groupbyKeyView,
            nullPolicy,
            requests,
            groupKeys,
            results,
            detourError)) {
      LOG(ERROR) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "cpuDecimalDetour.failure step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << requests.size() << " groupingKeys="
                 << numGroupingKeys << " error=" << detourError;
      VELOX_FAIL("HashAgg decimal CPU detour failed: {}", detourError);
    }
  } else {
    try {
      auto aggregateOutput = groupByOwner.aggregate(requests, stream);
      groupKeys = std::move(aggregateOutput.first);
      results = std::move(aggregateOutput.second);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[CudfHashAggDebug] stage=HashAgg.doGroupByAggregation."
                    "aggregateException step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << requests.size() << " groupingKeys="
                 << numGroupingKeys << " error=" << e.what();
      logCudaCheckpoint(
          "HashAgg.doGroupByAggregation.aggregateException.cudaCheckpoint",
          step_,
          stream,
          tableView.num_rows(),
          tableView.num_columns(),
          requests.size());
      throw;
    }
  }
  maybeDeviceSyncProbe(
      kDeviceSyncGroupGroupByCore,
      2,
      "HashAgg.doGroupByAggregation.postAggregate",
      step_,
      stream,
      groupKeys ? groupKeys->num_rows() : 0,
      results.size(),
      numGroupingKeys);
  logCudaCheckpoint(
      "HashAgg.doGroupByAggregation.afterAggregate.cudaCheckpoint",
      step_,
      stream,
      groupKeys ? groupKeys->num_rows() : 0,
      results.size(),
      numGroupingKeys);
  logHashAggDebug(
      "HashAgg.doGroupByAggregation.afterAggregate",
      step_,
      stream,
      groupKeys ? groupKeys->num_rows() : 0,
      results.size(),
      numGroupingKeys);
  logGroupbyResultsDebug(step_, stream, results);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  // first fill the grouping keys
  auto groupKeysColumns = groupKeys->release();
  resultColumns.insert(
      resultColumns.begin(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  // then fill the aggregation results
  maybeDeviceSyncProbe(
      kDeviceSyncGroupGroupByCore,
      3,
      "HashAgg.doGroupByAggregation.preMakeOutputColumns",
      step_,
      stream,
      groupKeys ? groupKeys->num_rows() : 0,
      results.size(),
      aggregators.size());
  for (size_t i = 0; i < aggregators.size(); ++i) {
    auto& aggregator = aggregators[i];
    logCudaCheckpoint(
        "HashAgg.doGroupByAggregation.beforeMakeOutputColumn.cudaCheckpoint",
        step_,
        stream,
        groupKeys ? groupKeys->num_rows() : 0,
        results.size(),
        i);
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream));
    if (resultColumns.back()) {
      logColumnViewDebug(
          "HashAgg.doGroupByAggregation.outputColumn",
          step_,
          stream,
          resultColumns.back()->view(),
          i);
    }
    logCudaCheckpoint(
        "HashAgg.doGroupByAggregation.afterMakeOutputColumn.cudaCheckpoint",
        step_,
        stream,
        groupKeys ? groupKeys->num_rows() : 0,
        results.size(),
        i);
  }
  maybeDeviceSyncProbe(
      kDeviceSyncGroupGroupByCore,
      4,
      "HashAgg.doGroupByAggregation.postMakeOutputColumns",
      step_,
      stream,
      groupKeys ? groupKeys->num_rows() : 0,
      resultColumns.size(),
      0);

  // make a cudf table out of columns
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));

  if (!dumpDir.empty()) {
    maybeWriteGroupbyOutputDump(
        dumpDir, step_, numGroupingKeys, stream, resultTable->view());
  }

  if (CudfConfig::getInstance().debugHashAggEndToEndValidate) {
    auto const& cfg = CudfConfig::getInstance();
    EndToEndValidationResult validation;
    if (!cfg.debugHashAggExpectedPath.empty()) {
      validation = validateOutputAgainstExpected(
          step_,
          stream,
          resultTable->view(),
          cfg.debugHashAggEndToEndBatchRows,
          cfg.debugHashAggExpectedPath);
      if (validation.skipped) {
        LOG(INFO) << "[HashAggEndToEnd] skipped reason=" << validation.reason
                  << " step=" << stepName(step_)
                  << " keyCols=" << groupbyKeyView.num_columns()
                  << " outputCols=" << resultTable->num_columns()
                  << " requestCount=" << requests.size();
      } else {
        LOG(INFO) << "[HashAggEndToEnd] expectedKeys=" << validation.expectedKeys
                  << " outputKeys=" << validation.outputKeys
                  << " checked=" << validation.checked
                  << " mismatches=" << validation.mismatches
                  << " unexpected=" << validation.missing;
        if (validation.mismatches > 0) {
          LOG(INFO) << "[HashAggEndToEnd] firstMismatch key="
                    << validation.firstKey
                    << " expected=" << toString128(validation.firstExpected)
                    << " actual=" << toString128(validation.firstActual);
        }
      }

      if (!validation.skipped && validation.mismatches > 0 &&
          (step_ == core::AggregationNode::Step::kFinal ||
           step_ == core::AggregationNode::Step::kSingle)) {
        auto inputValidation = validateEndToEndHashAgg(
            step_,
            stream,
            groupbyKeyView,
            nullPolicy,
            requests,
            resultTable->view(),
            cfg.debugHashAggEndToEndMaxRows,
            cfg.debugHashAggEndToEndBatchRows);
        if (inputValidation.skipped) {
          LOG(INFO) << "[HashAggEndToEndInput] skipped reason="
                    << inputValidation.reason << " step=" << stepName(step_)
                    << " keyCols=" << groupbyKeyView.num_columns()
                    << " outputCols=" << resultTable->num_columns()
                    << " requestCount=" << requests.size();
        } else {
          LOG(INFO) << "[HashAggEndToEndInput] expectedKeys="
                    << inputValidation.expectedKeys
                    << " outputKeys=" << inputValidation.outputKeys
                    << " checked=" << inputValidation.checked
                    << " mismatches=" << inputValidation.mismatches
                    << " missing=" << inputValidation.missing;
          if (inputValidation.mismatches > 0) {
            LOG(INFO) << "[HashAggEndToEndInput] firstMismatch key="
                      << inputValidation.firstKey
                      << " expected=" << toString128(inputValidation.firstExpected)
                      << " actual=" << toString128(inputValidation.firstActual);
          }
        }
      }
    } else {
      validation = validateEndToEndHashAgg(
          step_,
          stream,
          groupbyKeyView,
          nullPolicy,
          requests,
          resultTable->view(),
          cfg.debugHashAggEndToEndMaxRows,
          cfg.debugHashAggEndToEndBatchRows);
      if (validation.skipped) {
        LOG(INFO) << "[HashAggEndToEnd] skipped reason=" << validation.reason
                  << " step=" << stepName(step_)
                  << " keyCols=" << groupbyKeyView.num_columns()
                  << " outputCols=" << resultTable->num_columns()
                  << " requestCount=" << requests.size();
      } else {
        LOG(INFO) << "[HashAggEndToEnd] expectedKeys=" << validation.expectedKeys
                  << " outputKeys=" << validation.outputKeys
                  << " checked=" << validation.checked
                  << " mismatches=" << validation.mismatches
                  << " unexpected=" << validation.missing;
        if (validation.mismatches > 0) {
          LOG(INFO) << "[HashAggEndToEnd] firstMismatch key="
                    << validation.firstKey
                    << " expected=" << toString128(validation.firstExpected)
                    << " actual=" << toString128(validation.firstActual);
        }
      }
    }
  }

  auto numRows = resultTable->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfHashAggregation::doGlobalAggregation(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  logHashAggDebug(
      "HashAgg.doGlobalAggregation.begin",
      step_,
      stream,
      tableView.num_rows(),
      tableView.num_columns(),
      aggregators_.size());
  logCudaCheckpoint(
      "HashAgg.doGlobalAggregation.begin.cudaCheckpoint",
      step_,
      stream,
      tableView.num_rows(),
      tableView.num_columns(),
      aggregators_.size());
  logTableViewDebug("HashAgg.doGlobalAggregation.inputTable", step_, stream, tableView);
  auto const decimalCpuDetourEnabled = hashAggDebugDecimalCpuAggregateMode() != 0;
  auto const useDecimalCpuDetour =
      decimalCpuDetourEnabled && hasDecimalAggregator(aggregators_) &&
      tableView.num_rows() > 0;
  if (useDecimalCpuDetour) {
    LOG(WARNING) << "[CudfHashAggDebug] stage=HashAgg.doGlobalAggregation."
                    "cpuDecimalDetour.routeToGroupBy step="
                 << stepName(step_) << " stream="
                 << reinterpret_cast<const void*>(stream.value()) << " rows="
                 << tableView.num_rows() << " cols=" << tableView.num_columns()
                 << " requestCount=" << aggregators_.size();
    return doGroupByAggregation(tableView, {}, aggregators_, stream);
  }
  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(aggregators_.size());
  for (auto i = 0; i < aggregators_.size(); i++) {
    logHashAggDebug(
        "HashAgg.doGlobalAggregation.beforeDoReduce",
        step_,
        stream,
        tableView.num_rows(),
        tableView.num_columns(),
        i);
    logCudaCheckpoint(
        "HashAgg.doGlobalAggregation.beforeDoReduce.cudaCheckpoint",
        step_,
        stream,
        tableView.num_rows(),
        tableView.num_columns(),
        i);
    resultColumns.push_back(
        aggregators_[i]->doReduce(tableView, outputType_->childAt(i), stream));
    if (resultColumns.back()) {
      logColumnViewDebug(
          "HashAgg.doGlobalAggregation.doReduceOutput",
          step_,
          stream,
          resultColumns.back()->view(),
          i);
    }
    logCudaCheckpoint(
        "HashAgg.doGlobalAggregation.afterDoReduce.cudaCheckpoint",
        step_,
        stream,
        tableView.num_rows(),
        tableView.num_columns(),
        i);
  }
  logHashAggDebug(
      "HashAgg.doGlobalAggregation.end",
      step_,
      stream,
      1,
      resultColumns.size(),
      0);

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(resultColumns)),
      stream);
}

CudfVectorPtr CudfHashAggregation::getDistinctKeys(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    rmm::cuda_stream_view stream) {
  auto result = cudf::distinct(
      tableView.select(groupByKeys.begin(), groupByKeys.end()),
      {groupingKeyOutputChannels_.begin(), groupingKeyOutputChannels_.end()},
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream);

  auto numRows = result->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(result), stream);
}

CudfVectorPtr CudfHashAggregation::releaseAndResetPartialOutput() {
  VELOX_DCHECK(!isGlobal_);
  auto numOutputRows = partialOutput_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("flushRowCount", RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat("flushTimes", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "partialAggregationPct", RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving partialOutput_ to the caller because we want it to be null
  // after this call.
  return std::move(partialOutput_);
}

RowVectorPtr CudfHashAggregation::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (hashAggDebugEnabled()) {
    LOG(INFO) << "[CudfHashAggDebug] stage=getOutput.begin step="
              << stepName(step_) << " finished=" << finished_
              << " noMoreInput=" << noMoreInput_ << " bufferedInputs="
              << inputs_.size() << " hasPartialOutput="
              << static_cast<bool>(partialOutput_);
  }

  // Handle partial groupby and distinct.
  if (isPartialOutput_ && !isGlobal_) {
    if (partialOutput_ &&
        partialOutput_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      // This is basically a flush of the partial output.
      return releaseAndResetPartialOutput();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!partialOutput_ && finished_) {
      return nullptr;
    }
    return releaseAndResetPartialOutput();
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  logHashAggDebug(
      "HashAgg.getOutput.beforeConcat",
      step_,
      stream,
      inputs_.size(),
      0,
      0);
  maybeDeviceSyncProbe(
      kDeviceSyncGroupOperatorBoundary,
      5,
      "HashAgg.getOutput.beforeConcat",
      step_,
      stream,
      inputs_.size(),
      0,
      noMoreInput_ ? 1 : 0);

  auto tbl = getConcatenatedTable(inputs_, inputType_, stream);

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }
  maybeDeviceSyncProbe(
      kDeviceSyncGroupOperatorBoundary,
      6,
      "HashAgg.getOutput.afterInputRelease",
      step_,
      stream,
      tbl ? tbl->num_rows() : 0,
      tbl ? tbl->num_columns() : 0,
      static_cast<int64_t>(finished_));

  VELOX_CHECK_NOT_NULL(tbl);
  logHashAggDebug(
      "HashAgg.getOutput.afterConcat",
      step_,
      stream,
      tbl->num_rows(),
      tbl->num_columns(),
      0);
  maybeDeviceSyncProbe(
      kDeviceSyncGroupOperatorBoundary,
      1,
      "HashAgg.getOutput.afterConcat",
      step_,
      stream,
      tbl->num_rows(),
      tbl->num_columns(),
      noMoreInput_ ? 1 : 0);

  // Use tbl->view() instead of moving the table.
  // tbl stays alive until the end of this function, keeping the view valid.
  if (isDistinct_) {
    return getDistinctKeys(tbl->view(), groupingKeyInputChannels_, stream);
  } else if (isGlobal_) {
    return doGlobalAggregation(tbl->view(), stream);
  } else {
    return doGroupByAggregation(
        tbl->view(), groupingKeyInputChannels_, aggregators_, stream);
  }
}

void CudfHashAggregation::noMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfHashAggregation::isFinished() {
  return finished_;
}

// Step-aware aggregation registry implementation
StepAwareAggregationRegistry& getStepAwareAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

bool registerAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  auto& registry = getStepAwareAggregationRegistry();

  if (!overwrite && registry.find(name) != registry.end() &&
      registry[name].find(step) != registry[name].end()) {
    return false;
  }

  registry[name][step] = signatures;
  return true;
}

// Register step-aware builtin aggregation functions
bool registerStepAwareBuiltinAggregationFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // Register sum function (split by aggregation step)
  auto sumSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  // Decimal sum signatures.
  auto decimalSumSingle = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .returnType("decimal(38, a_scale)")
          .argumentType("decimal(a_precision, a_scale)")
          .build()};
  auto decimalSumPartial = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .returnType("varbinary")
          .argumentType("decimal(a_precision, a_scale)")
          .build()};
  auto decimalSumFinal = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_scale")
          .returnType("decimal(38, a_scale)")
          .argumentType("varbinary")
          .build()};
  auto decimalSumIntermediate = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("varbinary")
          .argumentType("varbinary")
          .build()};

  sumSingleSignatures.insert(
      sumSingleSignatures.end(),
      decimalSumSingle.begin(),
      decimalSumSingle.end());

  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      sumSingleSignatures);

  auto sumPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  sumPartialSignatures.insert(
      sumPartialSignatures.end(),
      decimalSumPartial.begin(),
      decimalSumPartial.end());

  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      sumPartialSignatures);

  auto sumFinalIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  auto sumFinalSignatures = sumFinalIntermediateSignatures;
  sumFinalSignatures.insert(
      sumFinalSignatures.end(),
      decimalSumFinal.begin(),
      decimalSumFinal.end());

  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kFinal,
      sumFinalSignatures);

  auto sumIntermediateSignatures = sumFinalIntermediateSignatures;
  sumIntermediateSignatures.insert(
      sumIntermediateSignatures.end(),
      decimalSumIntermediate.begin(),
      decimalSumIntermediate.end());

  registerAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kIntermediate,
      sumIntermediateSignatures);

  // Register count function (split by aggregation step)
  auto countSinglePartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("boolean")
          .build(),
      FunctionSignatureBuilder().returnType("bigint").build()};

  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kSingle,
      countSinglePartialSignatures);
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kPartial,
      countSinglePartialSignatures);

  auto countFinalIntermediateSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("bigint")
                                                  .argumentType("bigint")
                                                  .build()};
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kFinal,
      countFinalIntermediateSignatures);
  registerAggregationFunctionForStep(
      prefix + "count",
      core::AggregationNode::Step::kIntermediate,
      countFinalIntermediateSignatures);

  // Register min function (same signatures for all steps)
  auto minMaxSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("tinyint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("smallint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .integerVariable("p")
          .integerVariable("s")
          .returnType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .build()};

  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kSingle, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kPartial, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min", core::AggregationNode::Step::kFinal, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "min",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  // Register max function (same signatures for all steps)
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kSingle, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kPartial, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max", core::AggregationNode::Step::kFinal, minMaxSignatures);
  registerAggregationFunctionForStep(
      prefix + "max",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  // Register avg function (different signatures for different steps)

  // Single step: avg(input_type) -> double
  auto avgSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  // Decimal avg signatures.
  auto decimalAvgSingle = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .returnType("decimal(a_precision, a_scale)")
          .argumentType("decimal(a_precision, a_scale)")
          .build()};
  auto decimalAvgPartial = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .returnType("varbinary")
          .argumentType("decimal(a_precision, a_scale)")
          .build()};
  auto decimalAvgFinal = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .returnType("decimal(a_precision, a_scale)")
          .argumentType("varbinary")
          .build()};
  auto decimalAvgIntermediate = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("varbinary")
          .argumentType("varbinary")
          .build()};

  avgSingleSignatures.insert(
      avgSingleSignatures.end(),
      decimalAvgSingle.begin(),
      decimalAvgSingle.end());

  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      avgSingleSignatures);

  // Partial step: avg(input_type) -> row(sum input_type, count bigint)
  auto avgPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("double")
          .build()};

  avgPartialSignatures.insert(
      avgPartialSignatures.end(),
      decimalAvgPartial.begin(),
      decimalAvgPartial.end());

  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kPartial,
      avgPartialSignatures);

  // Final step: avg(row(double, bigint)) -> double
  auto avgFinalIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("row(double,bigint)")
          .build()};

  auto avgFinalSignatures = avgFinalIntermediateSignatures;
  avgFinalSignatures.insert(
      avgFinalSignatures.end(),
      decimalAvgFinal.begin(),
      decimalAvgFinal.end());

  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kFinal,
      avgFinalSignatures);

  // Intermediate step: avg(row(sum input_type, count bigint)) -> row(sum
  // input_type, count bigint)
  auto avgIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("row(double,bigint)")
          .build()};

  // WHY DOES SUM NOT HAVE THE EQUIVALENT OF THE ABOVE?
  // THE ABOVE THEN CLASHES WITH BELOW
  // @mattgara HELP! :)
  // auto avgIntermediateSignatures = avgFinalIntermediateSignatures;
  avgIntermediateSignatures.insert(
      avgIntermediateSignatures.end(),
      decimalAvgIntermediate.begin(),
      decimalAvgIntermediate.end());

  registerAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kIntermediate,
      avgIntermediateSignatures);

  return true;
}

bool matchTypedCallAgainstSignatures(
    const core::CallTypedExpr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call.inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& input : call.inputs()) {
    argTypes.push_back(input->type());
  }
  for (const auto& sig : sigs) {
    std::vector<Coercion> coercions(n);
    exec::SignatureBinder binder(*sig, argTypes);
    if (!binder.tryBindWithCoercions(coercions)) {
      continue;
    }

    // For simplicity we skip checking for constant agruments, this may be added
    // in the future

    return true;
  }
  return false;
}

// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  // Check against step-aware aggregation registry
  auto& stepAwareRegistry = getStepAwareAggregationRegistry();
  auto funcIt = stepAwareRegistry.find(call.name());
  if (funcIt == stepAwareRegistry.end()) {
    return false;
  }

  auto stepIt = funcIt->second.find(step);
  if (stepIt == funcIt->second.end()) {
    return false;
  }

  // Validate against step-specific signatures from registry
  return matchTypedCallAgainstSignatures(call, stepIt->second);
}

bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Optional detour: force CPU aggregation for decimal SUM/AVG inputs.
    // This avoids cuDF decimal groupby issues while the bug is investigated.
    const auto& rawTypes = aggregate.rawInputTypes;
    if (CudfConfig::getInstance().debugDisableDecimalSumAvgGpu &&
        rawTypes.size() == 1 && rawTypes[0] && rawTypes[0]->isDecimal()) {
      const auto& name = aggregate.call->name();
      const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
      if (name.rfind(prefix + "sum", 0) == 0 ||
          name.rfind(prefix + "avg", 0) == 0) {
        return false;
      }
    }
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    // `mask` is NOT supported (in testing do not appear to be be applied and
    // return incorrect results )
    if (aggregate.mask) {
      return false;
    }

    // Check input expressions can be evaluated by CUDF, expand the input first
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      std::vector<core::TypedExprPtr> exprs = {expandedInput};
      if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode.groupingKeys(), sourceNode, queryCtx)) {
    return false;
  }

  return true;
}

core::TypedExprPtr expandFieldReference(
    const core::TypedExprPtr& expr,
    const core::PlanNode* sourceNode) {
  // If this is a field reference and we have a source projection, expand it
  if (expr->kind() == core::ExprKind::kFieldAccess && sourceNode) {
    auto projectNode = dynamic_cast<const core::ProjectNode*>(sourceNode);
    if (projectNode) {
      auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr);
      if (fieldExpr) {
        // Find the corresponding projection expression
        const auto& projections = projectNode->projections();
        const auto& names = projectNode->names();
        for (size_t i = 0; i < names.size(); ++i) {
          if (names[i] == fieldExpr->name()) {
            return projections[i];
          }
        }
      }
    }
  }
  return expr;
}

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode,
    core::QueryCtx* queryCtx) {
  // Check grouping key expressions (with expansion)
  for (const auto& groupingKey : groupingKeys) {
    auto expandedKey = expandFieldReference(groupingKey, sourceNode);
    std::vector<core::TypedExprPtr> exprs = {expandedKey};
    if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
      return false;
    }
  }

  return true;
}

} // namespace facebook::velox::cudf_velox
