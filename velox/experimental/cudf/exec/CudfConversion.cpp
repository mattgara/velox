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

#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/VectorStream.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/Scratch.h"
#include "velox/serializers/PrestoSerializer.h"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/bit.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace facebook::velox::cudf_velox {

namespace {
struct HostColumnData {
  cudf::type_id typeId{cudf::type_id::EMPTY};
  int32_t scale{0};
  int64_t size{0};
  int64_t elementSize{0};
  int64_t nullCount{0};
  std::vector<uint8_t> data;
  std::vector<uint8_t> nullMask;
};

HostColumnData captureColumnHostData(
    rmm::cuda_stream_view stream,
    cudf::column_view const& column) {
  HostColumnData host;
  if (!cudf::is_fixed_width(column.type())) {
    throw std::runtime_error("unsupported non-fixed-width output column");
  }
  host.typeId = column.type().id();
  host.scale = column.type().scale();
  host.size = column.size();
  host.elementSize = cudf::size_of(column.type());
  host.nullCount = column.null_count();
  auto const dataBytes =
      static_cast<size_t>(host.size) * static_cast<size_t>(host.elementSize);
  host.data.resize(dataBytes);
  if (dataBytes > 0) {
    auto const* ptr = static_cast<uint8_t const*>(column.head()) +
        column.offset() * host.elementSize;
    auto const copyStatus = cudaMemcpyAsync(
        host.data.data(),
        ptr,
        dataBytes,
        cudaMemcpyDeviceToHost,
        stream.value());
    if (copyStatus != cudaSuccess) {
      throw std::runtime_error(
          "cudaMemcpyAsync output data failed: " +
          std::string(cudaGetErrorName(copyStatus)) + "(" +
          std::to_string(static_cast<int>(copyStatus)) + ")");
    }
  }
  if (host.nullCount > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(column.size()));
    host.nullMask.resize(maskBytes);
    if (maskBytes > 0) {
      auto const* maskPtr = column.null_mask();
      auto const copyStatus = cudaMemcpyAsync(
          host.nullMask.data(),
          maskPtr,
          maskBytes,
          cudaMemcpyDeviceToHost,
          stream.value());
      if (copyStatus != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpyAsync output null-mask failed: " +
            std::string(cudaGetErrorName(copyStatus)) + "(" +
            std::to_string(static_cast<int>(copyStatus)) + ")");
      }
    }
  }
  return host;
}

bool isValidAt(HostColumnData const& host, int64_t index) {
  if (host.nullCount == 0) {
    return true;
  }
  if (host.nullMask.empty()) {
    return false;
  }
  auto const* mask =
      reinterpret_cast<cudf::bitmask_type const*>(host.nullMask.data());
  return cudf::bit_is_set(mask, static_cast<cudf::size_type>(index));
}

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

bool readInt64Value(HostColumnData const& host, int64_t index, int64_t& out) {
  if (host.elementSize != static_cast<int64_t>(sizeof(int64_t))) {
    return false;
  }
  auto const* ptr = reinterpret_cast<int64_t const*>(host.data.data());
  out = ptr[index];
  return true;
}

bool readInt128Value(HostColumnData const& host, int64_t index, __int128_t& out) {
  if (host.elementSize != 16) {
    return false;
  }
  auto const* ptr = host.data.data() + index * 16;
  uint64_t lo = 0;
  int64_t hi = 0;
  std::memcpy(&lo, ptr, sizeof(uint64_t));
  std::memcpy(&hi, ptr + sizeof(uint64_t), sizeof(int64_t));
  out = (static_cast<__int128_t>(hi) << 64) | lo;
  return true;
}

void compareCudfToVeloxOutput(
    cudf::table_view const& tableView,
    RowVectorPtr const& output,
    RowTypePtr const& outputType,
    rmm::cuda_stream_view stream) {
  auto const& config = CudfConfig::getInstance();
  if (!config.debugCudfToVeloxValidate) {
    return;
  }
  if (!output || tableView.num_columns() == 0) {
    return;
  }

  auto numRows = static_cast<int64_t>(tableView.num_rows());
  if (config.debugCudfToVeloxMaxRows > 0 &&
      numRows > config.debugCudfToVeloxMaxRows) {
    auto slice = cudf::slice(
        tableView,
        {0, static_cast<cudf::size_type>(config.debugCudfToVeloxMaxRows)});
    if (!slice.empty()) {
      numRows = slice.front().num_rows();
      return compareCudfToVeloxOutput(
          slice.front(), output, outputType, stream);
    }
  }

  if (output->size() < numRows) {
    LOG(ERROR) << "[CudfToVeloxValidate] output row count smaller than cudf "
               << "rows outputRows=" << output->size()
               << " cudfRows=" << numRows;
    return;
  }

  if (output->childrenSize() != tableView.num_columns()) {
    LOG(ERROR) << "[CudfToVeloxValidate] column count mismatch outputCols="
               << output->childrenSize()
               << " cudfCols=" << tableView.num_columns();
    return;
  }

  std::vector<HostColumnData> hostColumns;
  hostColumns.reserve(tableView.num_columns());
  try {
    for (auto const& col : tableView) {
      hostColumns.push_back(captureColumnHostData(stream, col));
    }
    stream.synchronize();
  } catch (const std::exception& e) {
    LOG(ERROR) << "[CudfToVeloxValidate] capture failed: " << e.what();
    return;
  }

  SelectivityVector rows(numRows, true);
  std::vector<DecodedVector> decoded;
  decoded.reserve(output->childrenSize());
  for (size_t i = 0; i < output->childrenSize(); ++i) {
    decoded.emplace_back(*output->childAt(i), rows);
  }

  enum class CompareKind { kSkip, kInt64, kInt128 };
  std::vector<CompareKind> compareKinds;
  compareKinds.reserve(hostColumns.size());
  for (size_t colIdx = 0; colIdx < hostColumns.size(); ++colIdx) {
    auto const& type = outputType->childAt(colIdx);
    auto const& host = hostColumns[colIdx];
    if (type->isDecimal()) {
      auto scalePair = getDecimalPrecisionScale(*type);
      auto expectedScale = -static_cast<int32_t>(scalePair.second);
      if (host.scale != expectedScale) {
        LOG(ERROR) << "[CudfToVeloxValidate] scale mismatch col=" << colIdx
                   << " veloxScale=" << static_cast<int>(scalePair.second)
                   << " cudfScale=" << host.scale;
      }
      compareKinds.push_back(
          type->isLongDecimal() ? CompareKind::kInt128 : CompareKind::kInt64);
      continue;
    }
    if (type->kind() == TypeKind::BIGINT) {
      compareKinds.push_back(CompareKind::kInt64);
      continue;
    }
    compareKinds.push_back(CompareKind::kSkip);
  }

  int64_t mismatches = 0;
  int64_t nullMismatches = 0;
  int64_t checked = 0;
  int64_t firstRow = -1;
  size_t firstCol = 0;
  std::string firstCudf;
  std::string firstVelox;

  for (int64_t row = 0; row < numRows; ++row) {
    for (size_t colIdx = 0; colIdx < hostColumns.size(); ++colIdx) {
      if (compareKinds[colIdx] == CompareKind::kSkip) {
        continue;
      }
      auto const& host = hostColumns[colIdx];
      bool cudfValid = isValidAt(host, row);
      bool veloxValid = !decoded[colIdx].isNullAt(row);
      if (cudfValid != veloxValid) {
        ++nullMismatches;
        if (firstRow < 0) {
          firstRow = row;
          firstCol = colIdx;
          firstCudf = cudfValid ? "non-null" : "null";
          firstVelox = veloxValid ? "non-null" : "null";
        }
        continue;
      }
      if (!cudfValid) {
        continue;
      }

      if (compareKinds[colIdx] == CompareKind::kInt128) {
        __int128_t cudfValue{};
        if (!readInt128Value(host, row, cudfValue)) {
          continue;
        }
        auto veloxValue = decoded[colIdx].valueAt<int128_t>(row);
        ++checked;
        if (veloxValue != cudfValue) {
          ++mismatches;
          if (firstRow < 0) {
            firstRow = row;
            firstCol = colIdx;
            firstCudf = toString128(cudfValue);
            firstVelox = toString128(veloxValue);
          }
        }
      } else {
        int64_t cudfValue = 0;
        if (!readInt64Value(host, row, cudfValue)) {
          continue;
        }
        auto veloxValue = decoded[colIdx].valueAt<int64_t>(row);
        ++checked;
        if (veloxValue != cudfValue) {
          ++mismatches;
          if (firstRow < 0) {
            firstRow = row;
            firstCol = colIdx;
            firstCudf = std::to_string(cudfValue);
            firstVelox = std::to_string(veloxValue);
          }
        }
      }
    }
  }

  LOG(INFO) << "[CudfToVeloxValidate] rows=" << numRows
            << " cols=" << hostColumns.size() << " checked=" << checked
            << " mismatches=" << mismatches
            << " nullMismatches=" << nullMismatches;
  if (firstRow >= 0) {
    LOG(INFO) << "[CudfToVeloxValidate] firstMismatch row=" << firstRow
              << " col=" << firstCol << " cudf=" << firstCudf
              << " velox=" << firstVelox;
  }
}

void compareSerdeRoundtrip(
    RowVectorPtr const& output,
    RowTypePtr const& outputType,
    memory::MemoryPool* pool) {
  auto const& config = CudfConfig::getInstance();
  if (!config.debugSerdeValidate) {
    return;
  }
  if (!output || output->size() == 0) {
    return;
  }

  auto maxRows = static_cast<int64_t>(output->size());
  if (config.debugSerdeValidateMaxRows > 0 &&
      maxRows > config.debugSerdeValidateMaxRows) {
    maxRows = config.debugSerdeValidateMaxRows;
  }

  IndexRange range{0, static_cast<vector_size_t>(maxRows)};
  auto ranges = folly::Range<const IndexRange*>(&range, 1);

  serializer::presto::PrestoVectorSerde serde;
  serializer::presto::PrestoVectorSerde::PrestoOptions serdeOptions;
  auto batchSerializer = serde.createBatchSerializer(pool, &serdeOptions);

  std::ostringstream buffer;
  OStreamOutputStream out(&buffer);
  Scratch scratch;
  try {
    batchSerializer->serialize(output, ranges, scratch, &out);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[SerdeValidate] serialize failed: " << e.what();
    return;
  }

  auto data = buffer.str();
  if (data.empty()) {
    LOG(INFO) << "[SerdeValidate] empty serialized data";
    return;
  }
  ByteRange byteRange{
      reinterpret_cast<uint8_t*>(data.data()),
      static_cast<int64_t>(data.size()),
      0};
  BufferInputStream input({byteRange});

  RowVectorPtr roundTrip;
  try {
    serde.deserialize(&input, pool, outputType, &roundTrip, &serdeOptions);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[SerdeValidate] deserialize failed: " << e.what();
    return;
  }

  if (!roundTrip) {
    LOG(ERROR) << "[SerdeValidate] deserialize produced null vector";
    return;
  }

  SelectivityVector rows(maxRows, true);
  int64_t mismatches = 0;
  int64_t nullMismatches = 0;
  int64_t checked = 0;
  int64_t firstRow = -1;
  size_t firstCol = 0;
  std::string firstExpected;
  std::string firstActual;

  for (size_t colIdx = 0; colIdx < outputType->size(); ++colIdx) {
    auto const& type = outputType->childAt(colIdx);
    if (!type->isDecimal()) {
      continue;
    }
    DecodedVector left(*output->childAt(colIdx), rows);
    DecodedVector right(*roundTrip->childAt(colIdx), rows);
    for (int64_t row = 0; row < maxRows; ++row) {
      bool leftNull = left.isNullAt(row);
      bool rightNull = right.isNullAt(row);
      if (leftNull != rightNull) {
        ++nullMismatches;
        if (firstRow < 0) {
          firstRow = row;
          firstCol = colIdx;
          firstExpected = leftNull ? "null" : "non-null";
          firstActual = rightNull ? "null" : "non-null";
        }
        continue;
      }
      if (leftNull) {
        continue;
      }
      ++checked;
      if (type->isLongDecimal()) {
        auto expected = left.valueAt<int128_t>(row);
        auto actual = right.valueAt<int128_t>(row);
        if (expected != actual) {
          ++mismatches;
          if (firstRow < 0) {
            firstRow = row;
            firstCol = colIdx;
            firstExpected = toString128(expected);
            firstActual = toString128(actual);
          }
        }
      } else {
        auto expected = left.valueAt<int64_t>(row);
        auto actual = right.valueAt<int64_t>(row);
        if (expected != actual) {
          ++mismatches;
          if (firstRow < 0) {
            firstRow = row;
            firstCol = colIdx;
            firstExpected = std::to_string(expected);
            firstActual = std::to_string(actual);
          }
        }
      }
    }
  }

  LOG(INFO) << "[SerdeValidate] rows=" << maxRows << " checked=" << checked
            << " mismatches=" << mismatches
            << " nullMismatches=" << nullMismatches;
  if (firstRow >= 0) {
    LOG(INFO) << "[SerdeValidate] firstMismatch row=" << firstRow
              << " col=" << firstCol << " expected=" << firstExpected
              << " actual=" << firstActual;
  }
}

// Concatenate multiple RowVectors into a single RowVector.
// Copied from AggregationFuzzer.cpp.
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& results,
    velox::memory::MemoryPool* pool) {
  VELOX_NVTX_FUNC_RANGE();
  if (results.size() == 1) {
    return results[0];
  }
  vector_size_t totalCount = 0;
  for (const auto& result : results) {
    totalCount += result->size();
  }
  auto copy =
      BaseVector::create<RowVector>(results[0]->type(), totalCount, pool);
  auto copyCount = 0;
  for (const auto& result : results) {
    copy->copy(result.get(), copyCount, 0, result->size());
    copyCount += result->size();
  }
  return copy;
}

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfFromVelox"),
      NvtxHelper(
          nvtx3::rgb{255, 140, 0}, // Orange
          operatorId,
          fmt::format("[{}]", planNodeId)) {}

void CudfFromVelox::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Accumulate inputs
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
  }
}

RowVectorPtr CudfFromVelox::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  const auto targetOutputSize =
      preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());

  finished_ = noMoreInput_ && inputs_.empty();

  if (finished_ or
      (currentOutputSize_ < targetOutputSize and not noMoreInput_) or
      inputs_.empty()) {
    return nullptr;
  }

  // Select inputs that don't exceed the max vector size limit
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    if (totalSize + input->size() <= maxVectorSize) {
      selectedInputs.push_back(input);
      totalSize += input->size();
    } else {
      break;
    }
  }

  // Combine selected RowVectors into a single RowVector
  auto input = mergeRowVectors(selectedInputs, inputs_[0]->pool());

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;

  // Early return if no input
  if (input->size() == 0) {
    return nullptr;
  }

  // Get a stream from the global stream pool
  auto stream = cudfGlobalStreamPool().get_stream();

  // Convert RowVector to cudf table
  auto tbl = with_arrow::toCudfTable(input, input->pool(), stream);

  stream.synchronize();

  VELOX_CHECK_NOT_NULL(tbl);

  // Return a CudfVector that owns the cudf table
  const auto size = tbl->num_rows();
  return std::make_shared<CudfVector>(
      input->pool(), outputType_, size, std::move(tbl), stream);
}

void CudfFromVelox::close() {
  cudf::get_default_stream().synchronize();
  exec::Operator::close();
  inputs_.clear();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfToVelox"),
      NvtxHelper(
          nvtx3::rgb{148, 0, 211}, // Purple
          operatorId,
          fmt::format("[{}]", planNodeId)) {}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

std::optional<uint64_t> CudfToVelox::averageRowSize() {
  if (!averageRowSize_) {
    averageRowSize_ =
        inputs_.front()->estimateFlatSize() / inputs_.front()->size();
  }
  return averageRowSize_;
}

RowVectorPtr CudfToVelox::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_ || inputs_.empty()) {
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  // Get the target batch size
  const auto targetBatchSize = outputBatchRows(averageRowSize());
  auto stream = inputs_.front()->stream();

  // Process single input directly in these cases:
  // 1. In passthrough mode
  // 2. If we only have one input and it's smaller than or equal to the target
  // batch size
  if (isPassthroughMode() ||
      (inputs_.size() == 1 && inputs_.front()->size() <= targetBatchSize)) {
    // Move the CudfVector out to keep it alive while we use the view.
    // This avoids expensive materialization when constructed from packed_table.
    auto cudfVector = std::move(inputs_.front());
    inputs_.pop_front();

    auto tableView = cudfVector->getTableView();
    if (tableView.num_rows() == 0) {
      finished_ = noMoreInput_ && inputs_.empty();
      return nullptr;
    }
    RowVectorPtr output =
        with_arrow::toVeloxColumn(tableView, pool(), outputType_, "", stream);
    stream.synchronize();
    finished_ = noMoreInput_ && inputs_.empty();
    output->setType(outputType_);
    compareCudfToVeloxOutput(tableView, output, outputType_, stream);
    compareSerdeRoundtrip(output, outputType_, pool());
    // cudfVector goes out of scope here, freeing the GPU memory
    return output;
  }

  // Calculate how many tables we need to concatenate to reach the target batch
  // size and collect them in a vector
  std::vector<CudfVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;

  while (!inputs_.empty() && totalSize < targetBatchSize) {
    auto& input = inputs_.front();
    if (totalSize + input->size() <= targetBatchSize) {
      totalSize += input->size();
      selectedInputs.push_back(std::move(input));
      inputs_.pop_front();
    } else {
      // If the next input would exceed targetBatchSize,
      // we need to split it and only take what we need
      auto cudfTableView = input->getTableView();
      auto partitions = std::vector<cudf::size_type>{
          static_cast<cudf::size_type>(targetBatchSize - totalSize)};
      auto tableSplits = cudf::split(cudfTableView, partitions);

      // Create new CudfVector from the first part
      auto firstPart = std::make_unique<cudf::table>(tableSplits[0], stream);
      auto firstPartSize = firstPart->num_rows();
      auto firstPartVector = std::make_shared<CudfVector>(
          pool(), input->type(), firstPartSize, std::move(firstPart), stream);

      // Create new CudfVector from the second part
      auto secondPart = std::make_unique<cudf::table>(tableSplits[1], stream);
      auto secondPartSize = secondPart->num_rows();
      auto secondPartVector = std::make_shared<CudfVector>(
          pool(), input->type(), secondPartSize, std::move(secondPart), stream);

      // Replace the original input with the second part
      input = std::move(secondPartVector);

      // Add the first part to selectedInputs
      selectedInputs.push_back(std::move(firstPartVector));
      totalSize += firstPartSize;
      break;
    }
  }

  finished_ = noMoreInput_ && inputs_.empty();

  // If we have no inputs to process, return nullptr
  if (selectedInputs.empty()) {
    return nullptr;
  }

  // Concatenate the selected tables on the GPU
  auto resultTable = getConcatenatedTable(selectedInputs, outputType_, stream);

  // Convert the concatenated table to a RowVector
  const auto size = resultTable->num_rows();
  VELOX_CHECK_NOT_NULL(resultTable);
  if (size == 0) {
    return nullptr;
  }

  RowVectorPtr output =
      with_arrow::toVeloxColumn(resultTable->view(), pool(), outputType_, "", stream);
  stream.synchronize();
  finished_ = noMoreInput_ && inputs_.empty();
  output->setType(outputType_);
  compareCudfToVeloxOutput(resultTable->view(), output, outputType_, stream);
  compareSerdeRoundtrip(output, outputType_, pool());
  return output;
}

void CudfToVelox::close() {
  exec::Operator::close();
  inputs_.clear();
}

} // namespace facebook::velox::cudf_velox
