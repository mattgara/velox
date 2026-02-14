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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <string_view>
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

constexpr int32_t kDeviceSyncGroupGroupByCore{1};
constexpr int32_t kDeviceSyncGroupDecimalRequest{2};
constexpr int32_t kDeviceSyncGroupDecimalCompute{3};
constexpr int32_t kDeviceSyncGroupGlobalAgg{4};
constexpr int32_t kDeviceSyncGroupOperatorBoundary{5};

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
      << "3:decimal-compute,4:global-agg,5:operator-boundary";
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
      auto& request = requests.emplace_back();
      sumIdx_ = requests.size() - 1;
      request.values = tbl.column(inputIndex);
      request.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      if (step == core::AggregationNode::Step::kPartial ||
          (step == core::AggregationNode::Step::kSingle && isAvg_)) {
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
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
    if (isAvg_ && step == core::AggregationNode::Step::kSingle) {
      auto count = std::move(results[sumIdx_].results[1]);
      return computeAvgColumn(std::move(col), std::move(count), stream);
    }
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      if (count->type().id() != cudf::type_id::INT64) {
        count = cudf::cast(*count, cudf::data_type{cudf::type_id::INT64}, stream);
      }
      return cudf_velox::serializeDecimalSumState(
          col->view(), count->view(), stream);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      if (count->type().id() != cudf::type_id::INT64) {
        count = cudf::cast(*count, cudf::data_type{cudf::type_id::INT64}, stream);
      }
      return cudf_velox::serializeDecimalSumState(
          col->view(), count->view(), stream);
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

  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby groupByOwner(
      groupbyKeyView,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

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
  logCudaCheckpoint(
      "HashAgg.doGroupByAggregation.beforeAggregate.cudaCheckpoint",
      step_,
      stream,
      tableView.num_rows(),
      requests.size(),
      numGroupingKeys);
  std::unique_ptr<cudf::table> groupKeys;
  std::vector<cudf::groupby::aggregation_result> results;
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
