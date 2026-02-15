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

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ColumnSpec {
  int32_t typeId{0};
  int32_t scale{0};
  int64_t size{0};
  int64_t elementSize{0};
  int64_t nullCount{0};
  std::string dataFile;
  std::string nullMaskFile;
};

struct ColumnHostData {
  ColumnSpec spec;
  std::vector<uint8_t> data;
  std::vector<uint8_t> nullMask;
};

std::string trim(std::string value) {
  auto const begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  auto const end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::unordered_map<std::string, std::string> parseManifest(
    std::filesystem::path const& manifestPath) {
  std::ifstream in(manifestPath);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open manifest: " + manifestPath.string());
  }
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto const pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    auto key = trim(line.substr(0, pos));
    auto value = trim(line.substr(pos + 1));
    values[key] = value;
  }
  return values;
}

enum class DumpStep { kPartial, kIntermediate, kFinal, kSingle, kUnknown };

DumpStep parseDumpStep(std::string const& value) {
  if (value == "partial") {
    return DumpStep::kPartial;
  }
  if (value == "intermediate") {
    return DumpStep::kIntermediate;
  }
  if (value == "final") {
    return DumpStep::kFinal;
  }
  if (value == "single") {
    return DumpStep::kSingle;
  }
  return DumpStep::kUnknown;
}

std::string requireValue(
    std::unordered_map<std::string, std::string> const& manifest,
    std::string const& key) {
  auto it = manifest.find(key);
  if (it == manifest.end()) {
    throw std::runtime_error("missing manifest key: " + key);
  }
  return it->second;
}

DumpStep requireStep(
    std::unordered_map<std::string, std::string> const& manifest,
    std::string const& key) {
  return parseDumpStep(requireValue(manifest, key));
}

int32_t requireInt32(
    std::unordered_map<std::string, std::string> const& manifest,
    std::string const& key) {
  return static_cast<int32_t>(std::stoll(requireValue(manifest, key)));
}

int64_t requireInt64(
    std::unordered_map<std::string, std::string> const& manifest,
    std::string const& key) {
  return std::stoll(requireValue(manifest, key));
}

std::vector<uint8_t> readBinaryFile(std::filesystem::path const& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  in.seekg(0, std::ios::end);
  auto const size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!in) {
      throw std::runtime_error("failed reading binary file: " + path.string());
    }
  }
  return data;
}

ColumnSpec parseColumnSpec(
    std::unordered_map<std::string, std::string> const& manifest,
    std::string const& prefix) {
  ColumnSpec spec;
  spec.typeId = requireInt32(manifest, prefix + ".type_id");
  spec.scale = requireInt32(manifest, prefix + ".scale");
  spec.size = requireInt64(manifest, prefix + ".size");
  spec.elementSize = requireInt64(manifest, prefix + ".element_size");
  spec.nullCount = requireInt64(manifest, prefix + ".null_count");
  spec.dataFile = requireValue(manifest, prefix + ".data_file");
  auto it = manifest.find(prefix + ".null_mask_file");
  spec.nullMaskFile = (it == manifest.end()) ? "" : it->second;
  return spec;
}

ColumnHostData readColumnHostData(
    std::filesystem::path const& dumpDir,
    ColumnSpec const& spec) {
  ColumnHostData host;
  host.spec = spec;
  host.data = readBinaryFile(dumpDir / spec.dataFile);
  auto const expectedBytes =
      static_cast<size_t>(spec.size) * static_cast<size_t>(spec.elementSize);
  if (host.data.size() != expectedBytes) {
    throw std::runtime_error(
        "data size mismatch for " + spec.dataFile + ": expected " +
        std::to_string(expectedBytes) + " got " + std::to_string(host.data.size()));
  }
  if (spec.nullCount > 0) {
    if (spec.nullMaskFile.empty()) {
      throw std::runtime_error(
          "null_count > 0 but null mask file missing for " + spec.dataFile);
    }
    host.nullMask = readBinaryFile(dumpDir / spec.nullMaskFile);
    auto const expectedMaskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(spec.size));
    if (host.nullMask.size() != expectedMaskBytes) {
      throw std::runtime_error(
          "null-mask size mismatch for " + spec.nullMaskFile + ": expected " +
          std::to_string(expectedMaskBytes) + " got " +
          std::to_string(host.nullMask.size()));
    }
  }
  return host;
}

std::unique_ptr<cudf::column> makeColumnFromHostData(
    ColumnHostData const& host,
    rmm::cuda_stream_view stream) {
  auto const size = static_cast<cudf::size_type>(host.spec.size);
  auto const type = cudf::data_type{
      static_cast<cudf::type_id>(host.spec.typeId), host.spec.scale};
  auto col = cudf::make_fixed_width_column(
      type, size, cudf::mask_state::UNALLOCATED, stream);

  auto const expectedBytes =
      static_cast<size_t>(host.spec.size) * static_cast<size_t>(host.spec.elementSize);
  if (expectedBytes > 0) {
    auto const copyStatus = cudaMemcpyAsync(
        col->mutable_view().data<uint8_t>(),
        host.data.data(),
        expectedBytes,
        cudaMemcpyHostToDevice,
        stream.value());
    if (copyStatus != cudaSuccess) {
      throw std::runtime_error(
          "cudaMemcpyAsync data failed: " +
          std::string(cudaGetErrorName(copyStatus)) + "(" +
          std::to_string(static_cast<int>(copyStatus)) + ")");
    }
  }

  if (host.spec.nullCount > 0) {
    auto const expectedMaskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(size));
    rmm::device_buffer maskBuffer(expectedMaskBytes, stream);
    if (expectedMaskBytes > 0) {
      auto const copyStatus = cudaMemcpyAsync(
          maskBuffer.data(),
          host.nullMask.data(),
          expectedMaskBytes,
          cudaMemcpyHostToDevice,
          stream.value());
      if (copyStatus != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpyAsync null-mask failed: " +
            std::string(cudaGetErrorName(copyStatus)) + "(" +
            std::to_string(static_cast<int>(copyStatus)) + ")");
      }
    }
    col->set_null_mask(
        std::move(maskBuffer),
        static_cast<cudf::size_type>(host.spec.nullCount));
  }
  stream.synchronize();
  return col;
}

std::unique_ptr<cudf::groupby_aggregation> makeGroupbyAggregation(
    cudf::aggregation::Kind kind) {
  using Kind = cudf::aggregation::Kind;
  switch (kind) {
    case Kind::SUM:
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case Kind::MIN:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case Kind::MAX:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case Kind::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(
          cudf::null_policy::EXCLUDE);
    case Kind::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(
          cudf::null_policy::INCLUDE);
    case Kind::MEAN:
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::runtime_error(
          "unsupported aggregation kind: " +
          std::to_string(static_cast<int32_t>(kind)));
  }
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

struct Int128KeyHash {
  size_t operator()(const __int128_t& value) const {
    uint64_t low = static_cast<uint64_t>(value);
    uint64_t high = static_cast<uint64_t>(value >> 64);
    size_t seed = std::hash<uint64_t>{}(low);
    seed ^= std::hash<uint64_t>{}(high) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
        (seed >> 2);
    return seed;
  }
};

bool isSupportedAggKind(cudf::aggregation::Kind kind) {
  return kind == cudf::aggregation::Kind::SUM ||
      kind == cudf::aggregation::Kind::COUNT_VALID ||
      kind == cudf::aggregation::Kind::COUNT_ALL;
}

struct SumCountIndices {
  int sumIdx{-1};
  int countIdx{-1};
};

SumCountIndices findSumCountIndices(
    std::vector<cudf::aggregation::Kind> const& kinds) {
  SumCountIndices indices;
  for (size_t i = 0; i < kinds.size(); ++i) {
    if (kinds[i] == cudf::aggregation::Kind::SUM && indices.sumIdx < 0) {
      indices.sumIdx = static_cast<int>(i);
    } else if ((kinds[i] == cudf::aggregation::Kind::COUNT_VALID ||
                kinds[i] == cudf::aggregation::Kind::COUNT_ALL) &&
               indices.countIdx < 0) {
      indices.countIdx = static_cast<int>(i);
    }
  }
  return indices;
}

bool decodeValueInt128(
    cudf::type_id type,
    const uint8_t* data,
    int64_t index,
    int64_t elementSize,
    __int128_t& out) {
  auto const* ptr = data + static_cast<size_t>(index) * elementSize;
  switch (type) {
    case cudf::type_id::DECIMAL128: {
      __int128_t value{};
      std::memcpy(&value, ptr, sizeof(__int128_t));
      out = value;
      return true;
    }
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::INT64: {
      int64_t value{};
      std::memcpy(&value, ptr, sizeof(int64_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::INT32: {
      int32_t value{};
      std::memcpy(&value, ptr, sizeof(int32_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::INT16: {
      int16_t value{};
      std::memcpy(&value, ptr, sizeof(int16_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::INT8: {
      int8_t value{};
      std::memcpy(&value, ptr, sizeof(int8_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::UINT64: {
      uint64_t value{};
      std::memcpy(&value, ptr, sizeof(uint64_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::UINT32: {
      uint32_t value{};
      std::memcpy(&value, ptr, sizeof(uint32_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::UINT16: {
      uint16_t value{};
      std::memcpy(&value, ptr, sizeof(uint16_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    case cudf::type_id::UINT8: {
      uint8_t value{};
      std::memcpy(&value, ptr, sizeof(uint8_t));
      out = static_cast<__int128_t>(value);
      return true;
    }
    default:
      return false;
  }
}

bool decodeValueInt128(
    ColumnHostData const& host,
    int64_t index,
    __int128_t& out) {
  return decodeValueInt128(
      static_cast<cudf::type_id>(host.spec.typeId),
      host.data.data(),
      index,
      host.spec.elementSize,
      out);
}

bool isValidAt(ColumnHostData const& host, int64_t index) {
  if (host.spec.nullCount == 0) {
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

ColumnHostData readOutputColumnHostData(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream) {
  ColumnHostData host;
  host.spec.typeId = static_cast<int32_t>(col.type().id());
  host.spec.scale = col.type().scale();
  host.spec.size = col.size();
  host.spec.elementSize = cudf::size_of(col.type());
  host.spec.nullCount = col.null_count();
  auto const dataBytes = static_cast<size_t>(host.spec.size) *
      static_cast<size_t>(host.spec.elementSize);
  host.data.resize(dataBytes);
  if (dataBytes > 0) {
    auto const* ptr = static_cast<uint8_t const*>(col.head()) +
        col.offset() * host.spec.elementSize;
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
  if (host.spec.nullCount > 0) {
    auto const maskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(col.size()));
    host.nullMask.resize(maskBytes);
    if (maskBytes > 0) {
      auto const* maskPtr = col.null_mask();
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
  stream.synchronize();
  return host;
}

std::filesystem::path parseManifestPath(int argc, char** argv) {
  std::filesystem::path manifestPath;
  std::filesystem::path sessionDir;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--session" && i + 1 < argc) {
      sessionDir = argv[++i];
      continue;
    }
    if (arg == "--manifest" && i + 1 < argc) {
      manifestPath = argv[++i];
      continue;
    }
    if (arg == "--dump_dir" && i + 1 < argc) {
      manifestPath = std::filesystem::path(argv[++i]) / "manifest.txt";
      continue;
    }
  }
  if (!sessionDir.empty()) {
    return sessionDir;
  }
  if (manifestPath.empty()) {
    throw std::runtime_error(
        "usage: velox_cudf_hashagg_dump_replay --manifest <path/to/manifest.txt> "
        "or --dump_dir <path/to/dump_dir> or --session <path/to/session_dir> "
        "[--replay-chain] [--no-validate] [--validate-max-rows N]");
  }
  return manifestPath;
}

struct ValidationOptions {
  bool enabled{true};
  int64_t maxRows{0};
};

struct ReplayOptions {
  bool replayChain{false};
};

ValidationOptions parseValidationOptions(int argc, char** argv) {
  ValidationOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--no-validate") {
      options.enabled = false;
    } else if (arg == "--validate-max-rows" && i + 1 < argc) {
      options.maxRows = std::stoll(argv[++i]);
    }
  }
  return options;
}

ReplayOptions parseReplayOptions(int argc, char** argv) {
  ReplayOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--replay-chain") {
      options.replayChain = true;
    }
  }
  return options;
}

std::vector<std::filesystem::path> readSessionIndex(
    std::filesystem::path const& sessionDir) {
  auto const indexPath = sessionDir / "hashagg_dump_index.txt";
  std::ifstream in(indexPath);
  if (!in.is_open()) {
    throw std::runtime_error(
        "failed to open session index: " + indexPath.string());
  }
  std::vector<std::filesystem::path> dumps;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    dumps.emplace_back(line);
  }
  return dumps;
}

int runChainSession(
    std::filesystem::path const& sessionDir,
    ValidationOptions const& validateOptions) {
  auto dumps = readSessionIndex(sessionDir);
  if (dumps.empty()) {
    throw std::runtime_error("session index is empty");
  }

  struct DumpInfo {
    std::filesystem::path manifestPath;
    DumpStep step{DumpStep::kUnknown};
    int32_t keyCount{0};
    int32_t requestCount{0};
    int32_t nullPolicyValue{0};
    int32_t scale{0};
    std::vector<cudf::aggregation::Kind> aggKinds;
  };

  std::vector<DumpInfo> partials;
  for (auto const& dumpDir : dumps) {
    auto manifestPath = dumpDir / "manifest.txt";
    auto manifest = parseManifest(manifestPath);
    DumpInfo info;
    info.manifestPath = manifestPath;
    info.step = requireStep(manifest, "step");
    info.keyCount = requireInt32(manifest, "key_count");
    info.requestCount = requireInt32(manifest, "request_count");
    info.nullPolicyValue = requireInt32(manifest, "null_policy");
    info.scale = requireInt32(manifest, "request.0.scale");
    auto aggCount = requireInt32(manifest, "request.0.aggregation_count");
    info.aggKinds.reserve(aggCount);
    for (int32_t aggIdx = 0; aggIdx < aggCount; ++aggIdx) {
      auto const kindValue = requireInt32(
          manifest, "request.0.aggregation_kind." + std::to_string(aggIdx));
      info.aggKinds.push_back(static_cast<cudf::aggregation::Kind>(kindValue));
    }
    if (info.step == DumpStep::kPartial || info.step == DumpStep::kSingle) {
      partials.push_back(std::move(info));
    }
  }

  if (partials.empty()) {
    throw std::runtime_error("no partial or single dumps found in session");
  }

  auto const stream = cudf::get_default_stream();
  std::unordered_map<__int128_t, __int128_t, Int128KeyHash> expectedSum;
  expectedSum.reserve(1000000);

  std::unique_ptr<cudf::table> partialOutput;
  int32_t scale = partials.front().scale;
  cudf::null_policy nullPolicy =
      static_cast<cudf::null_policy>(partials.front().nullPolicyValue);

  for (auto const& info : partials) {
    auto manifest = parseManifest(info.manifestPath);
    auto dumpDir = info.manifestPath.parent_path();
    auto keySpec = parseColumnSpec(manifest, "key.0");
    auto valueSpec = parseColumnSpec(manifest, "request.0");
    auto keyHost = readColumnHostData(dumpDir, keySpec);
    auto valueHost = readColumnHostData(dumpDir, valueSpec);

    if (validateOptions.maxRows > 0 &&
        keyHost.spec.size > validateOptions.maxRows) {
      throw std::runtime_error("input rows exceed validate-max-rows");
    }

    // CPU expected accumulation.
    for (int64_t row = 0; row < keyHost.spec.size; ++row) {
      if (keyHost.spec.nullCount > 0 && !isValidAt(keyHost, row)) {
        if (nullPolicy == cudf::null_policy::EXCLUDE) {
          continue;
        }
        throw std::runtime_error("null key encountered in expected sum");
      }
      __int128_t keyValue{};
      if (!decodeValueInt128(keyHost, row, keyValue)) {
        throw std::runtime_error("unsupported key type in expected sum");
      }
      if (valueHost.spec.nullCount > 0 && !isValidAt(valueHost, row)) {
        continue;
      }
      __int128_t value{};
      if (!decodeValueInt128(valueHost, row, value)) {
        throw std::runtime_error("unsupported value type in expected sum");
      }
      expectedSum[keyValue] += value;
    }

    // GPU groupby for this batch.
    std::vector<std::unique_ptr<cudf::column>> keyColumns;
    std::vector<cudf::column_view> keyViews;
    auto keyCol = makeColumnFromHostData(keyHost, stream);
    keyViews.push_back(keyCol->view());
    keyColumns.push_back(std::move(keyCol));
    cudf::table_view groupbyKeys(keyViews);

    std::vector<std::unique_ptr<cudf::column>> requestColumns;
    std::vector<cudf::groupby::aggregation_request> requests;
    auto valueCol = makeColumnFromHostData(valueHost, stream);
    requestColumns.push_back(std::move(valueCol));
    cudf::groupby::aggregation_request request;
    request.values = requestColumns.back()->view();
    for (auto kind : info.aggKinds) {
      request.aggregations.push_back(makeGroupbyAggregation(kind));
    }
    requests.push_back(std::move(request));

    cudf::groupby::groupby groupby(groupbyKeys, nullPolicy);
    auto output = groupby.aggregate(requests, stream);
    auto groupKeys = std::move(output.first);
    auto results = std::move(output.second);
    if (!groupKeys || results.empty()) {
      throw std::runtime_error("empty groupby output");
    }

    auto indices = findSumCountIndices(info.aggKinds);
    if (indices.sumIdx < 0 || indices.countIdx < 0) {
      throw std::runtime_error("missing SUM/COUNT in partial aggregation");
    }
    auto sumCol = std::move(results[0].results[indices.sumIdx]);
    auto countCol = std::move(results[0].results[indices.countIdx]);

    auto stateCol = facebook::velox::cudf_velox::serializeDecimalSumState(
        sumCol->view(), countCol->view(), stream);

    auto keyCols = groupKeys->release();
    keyCols.push_back(std::move(stateCol));
    auto batchTable = std::make_unique<cudf::table>(std::move(keyCols));

    if (partialOutput) {
      std::vector<cudf::table_view> tablesToConcat;
      tablesToConcat.push_back(partialOutput->view());
      tablesToConcat.push_back(batchTable->view());
      auto concatenated = cudf::concatenate(tablesToConcat, stream);

      auto stateView = concatenated->view().column(1);
      auto decoded = facebook::velox::cudf_velox::deserializeDecimalSumStateWithCount(
          stateView, scale, stream);
      std::vector<cudf::column_view> mergeKeyViews;
      mergeKeyViews.push_back(concatenated->view().column(0));
      cudf::table_view mergeKeys(mergeKeyViews);

      std::vector<cudf::groupby::aggregation_request> mergeRequests;
      cudf::groupby::aggregation_request sumRequest;
      sumRequest.values = decoded.sum->view();
      sumRequest.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      mergeRequests.push_back(std::move(sumRequest));

      cudf::groupby::aggregation_request countRequest;
      countRequest.values = decoded.count->view();
      countRequest.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      mergeRequests.push_back(std::move(countRequest));

      cudf::groupby::groupby mergeGroupby(mergeKeys, nullPolicy);
      auto mergeOutput = mergeGroupby.aggregate(mergeRequests, stream);
      auto mergeKeysOut = std::move(mergeOutput.first);
      auto mergeResults = std::move(mergeOutput.second);
      auto mergedSum = std::move(mergeResults[0].results[0]);
      auto mergedCount = std::move(mergeResults[1].results[0]);
      auto mergedState = facebook::velox::cudf_velox::serializeDecimalSumState(
          mergedSum->view(), mergedCount->view(), stream);

      auto mergedKeyCols = mergeKeysOut->release();
      mergedKeyCols.push_back(std::move(mergedState));
      partialOutput = std::make_unique<cudf::table>(std::move(mergedKeyCols));
    } else {
      partialOutput = std::move(batchTable);
    }
  }

  if (!partialOutput) {
    throw std::runtime_error("no partial output generated");
  }

  auto finalStateView = partialOutput->view().column(1);
  auto finalDecoded = facebook::velox::cudf_velox::deserializeDecimalSumStateWithCount(
      finalStateView, scale, stream);
  std::vector<cudf::column_view> finalKeyViews;
  finalKeyViews.push_back(partialOutput->view().column(0));
  cudf::table_view finalKeys(finalKeyViews);

  std::vector<cudf::groupby::aggregation_request> finalRequests;
  cudf::groupby::aggregation_request finalSumRequest;
  finalSumRequest.values = finalDecoded.sum->view();
  finalSumRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  finalRequests.push_back(std::move(finalSumRequest));

  cudf::groupby::groupby finalGroupby(finalKeys, nullPolicy);
  auto finalOutput = finalGroupby.aggregate(finalRequests, stream);
  auto finalKeysOut = std::move(finalOutput.first);
  auto finalResults = std::move(finalOutput.second);

  auto keyHost = readOutputColumnHostData(finalKeysOut->view().column(0), stream);
  auto sumHost =
      readOutputColumnHostData(finalResults[0].results[0]->view(), stream);

  int64_t mismatches = 0;
  __int128_t firstMismatchKey = 0;
  __int128_t firstExpected = 0;
  __int128_t firstActual = 0;
  for (int64_t row = 0; row < keyHost.spec.size; ++row) {
    __int128_t keyValue{};
    __int128_t outValue{};
    if (!decodeValueInt128(keyHost, row, keyValue) ||
        !decodeValueInt128(sumHost, row, outValue)) {
      throw std::runtime_error("unsupported output types in final compare");
    }
    auto it = expectedSum.find(keyValue);
    __int128_t expectedValue = (it != expectedSum.end()) ? it->second : 0;
    if (outValue != expectedValue) {
      if (mismatches == 0) {
        firstMismatchKey = keyValue;
        firstExpected = expectedValue;
        firstActual = outValue;
      }
      ++mismatches;
    }
  }
  if (static_cast<int64_t>(expectedSum.size()) != keyHost.spec.size) {
    auto diff = static_cast<int64_t>(expectedSum.size()) - keyHost.spec.size;
    mismatches += diff >= 0 ? diff : -diff;
  }

  std::cout << "[HashAggDumpReplay] chain_validate_mismatches " << mismatches;
  if (mismatches > 0) {
    std::cout << " firstMismatchKey=" << toString128(firstMismatchKey)
              << " expected=" << toString128(firstExpected)
              << " actual=" << toString128(firstActual);
  }
  std::cout << std::endl;
  return mismatches > 0 ? 2 : 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    auto const inputPath = parseManifestPath(argc, argv);
    auto const validateOptions = parseValidationOptions(argc, argv);
    auto const replayOptions = parseReplayOptions(argc, argv);
    auto runOne = [&](std::filesystem::path const& manifestPath) -> int {
      bool validate = validateOptions.enabled;
      auto const dumpDir = manifestPath.parent_path();
      auto const manifest = parseManifest(manifestPath);
      auto const stream = cudf::get_default_stream();

      auto const keyCount = requireInt32(manifest, "key_count");
      auto const requestCount = requireInt32(manifest, "request_count");
      auto const nullPolicyValue = requireInt32(manifest, "null_policy");
      auto const nullPolicy = static_cast<cudf::null_policy>(nullPolicyValue);

      std::vector<std::unique_ptr<cudf::column>> keyColumns;
      std::vector<cudf::column_view> keyViews;
      std::vector<ColumnHostData> keyHostData;
      keyColumns.reserve(keyCount);
      keyViews.reserve(keyCount);
      keyHostData.reserve(keyCount);
      for (int32_t i = 0; i < keyCount; ++i) {
        auto spec = parseColumnSpec(manifest, "key." + std::to_string(i));
        auto host = readColumnHostData(dumpDir, spec);
        auto col = makeColumnFromHostData(host, stream);
        keyViews.push_back(col->view());
        keyColumns.push_back(std::move(col));
        keyHostData.push_back(std::move(host));
      }
      cudf::table_view groupbyKeys(keyViews);

      std::vector<std::unique_ptr<cudf::column>> requestValueColumns;
      std::vector<cudf::groupby::aggregation_request> requests;
      std::vector<ColumnHostData> requestHostData;
      std::vector<std::vector<cudf::aggregation::Kind>> requestAggKinds;
      requestValueColumns.reserve(requestCount);
      requests.reserve(requestCount);
      requestHostData.reserve(requestCount);
      requestAggKinds.reserve(requestCount);
      for (int32_t i = 0; i < requestCount; ++i) {
        auto const prefix = "request." + std::to_string(i);
        auto spec = parseColumnSpec(manifest, prefix);
        auto host = readColumnHostData(dumpDir, spec);
        auto col = makeColumnFromHostData(host, stream);
        requestValueColumns.push_back(std::move(col));
        requestHostData.push_back(std::move(host));

        cudf::groupby::aggregation_request request;
        request.values = requestValueColumns.back()->view();
        auto const aggCount = requireInt32(manifest, prefix + ".aggregation_count");
        request.aggregations.reserve(aggCount);
        std::vector<cudf::aggregation::Kind> aggKinds;
        aggKinds.reserve(aggCount);
        for (int32_t aggIdx = 0; aggIdx < aggCount; ++aggIdx) {
          auto const kindValue = requireInt32(
              manifest, prefix + ".aggregation_kind." + std::to_string(aggIdx));
          auto const kind = static_cast<cudf::aggregation::Kind>(kindValue);
          request.aggregations.push_back(makeGroupbyAggregation(kind));
          aggKinds.push_back(kind);
        }
        requests.push_back(std::move(request));
        requestAggKinds.push_back(std::move(aggKinds));
      }

      std::cout << "[HashAggDumpReplay] manifest=" << manifestPath.string()
                << " keyCount=" << keyCount << " requestCount=" << requestCount
                << " nullPolicy=" << nullPolicyValue
                << " validate=" << (validate ? 1 : 0) << std::endl;

      bool aggregateThrew = false;
      std::string exceptionText;
      int64_t outputRows = -1;
      int64_t outputCount = -1;
      std::unique_ptr<cudf::table> outputKeys;
      std::vector<cudf::groupby::aggregation_result> outputResults;
      try {
        cudf::groupby::groupby groupby(groupbyKeys, nullPolicy);
        auto output = groupby.aggregate(requests, stream);
        outputRows = output.first ? output.first->num_rows() : 0;
        outputCount = output.second.size();
        outputKeys = std::move(output.first);
        outputResults = std::move(output.second);
      } catch (const std::exception& e) {
        aggregateThrew = true;
        exceptionText = e.what();
      }

      auto const peekErr = cudaPeekAtLastError();
      auto const streamSyncErr = cudaStreamSynchronize(stream.value());
      auto const deviceSyncErr = cudaDeviceSynchronize();

      std::cout << "[HashAggDumpReplay] aggregateThrew="
                << (aggregateThrew ? 1 : 0)
                << " outputRows=" << outputRows
                << " outputCount=" << outputCount
                << " cudaPeek=" << cudaGetErrorName(peekErr) << "("
                << static_cast<int>(peekErr) << ")"
                << " cudaStreamSync=" << cudaGetErrorName(streamSyncErr) << "("
                << static_cast<int>(streamSyncErr) << ")"
                << " cudaDeviceSync=" << cudaGetErrorName(deviceSyncErr) << "("
                << static_cast<int>(deviceSyncErr) << ")" << std::endl;
      if (aggregateThrew) {
        std::cout << "[HashAggDumpReplay] exception=" << exceptionText << std::endl;
      }

      bool validationSkipped = false;
      std::string validationSkipReason;
      int64_t validationMismatches = 0;
      __int128_t firstMismatchKey = 0;
      __int128_t firstExpected = 0;
      __int128_t firstActual = 0;
      if (validate && !aggregateThrew) {
        if (validateOptions.maxRows > 0 &&
            !keyHostData.empty() &&
            keyHostData[0].spec.size > validateOptions.maxRows) {
          validationSkipped = true;
          validationSkipReason = "input rows exceed validate-max-rows";
        } else if (keyCount != 1) {
          validationSkipped = true;
          validationSkipReason = "validation requires single key column";
        } else if (requestCount != 1) {
          validationSkipped = true;
          validationSkipReason = "validation requires single request column";
        } else if (keyHostData[0].spec.nullCount > 0 &&
                   nullPolicy == cudf::null_policy::INCLUDE) {
          validationSkipped = true;
          validationSkipReason = "validation does not support null keys";
        }

        if (!validationSkipped) {
        bool hasSum = false;
        for (auto kind : requestAggKinds[0]) {
          if (kind == cudf::aggregation::Kind::SUM) {
            hasSum = true;
          }
        }
        if (!hasSum) {
          validationSkipped = true;
          validationSkipReason = "validation requires SUM aggregation";
        }
      }

      if (!validationSkipped) {
        for (auto kind : requestAggKinds[0]) {
          if (!isSupportedAggKind(kind)) {
            validationSkipped = true;
            validationSkipReason =
                std::string("unsupported aggregation kind: ") +
                aggregationKindName(kind);
            break;
          }
        }
      }

      if (!validationSkipped) {
        std::unordered_map<__int128_t, __int128_t, Int128KeyHash> expectedSum;
        std::unordered_map<__int128_t, int64_t, Int128KeyHash> expectedCount;
        std::unordered_map<__int128_t, int64_t, Int128KeyHash> expectedCountAll;
          auto const numRows = keyHostData[0].spec.size;
        expectedSum.reserve(static_cast<size_t>(numRows));
        expectedCount.reserve(static_cast<size_t>(numRows));
        expectedCountAll.reserve(static_cast<size_t>(numRows));
          for (int64_t row = 0; row < numRows; ++row) {
            if (keyHostData[0].spec.nullCount > 0 &&
                !isValidAt(keyHostData[0], row)) {
              if (nullPolicy == cudf::null_policy::EXCLUDE) {
                continue;
              }
              validationSkipped = true;
              validationSkipReason = "null key encountered";
              break;
            }
            __int128_t keyValue{};
            if (!decodeValueInt128(keyHostData[0], row, keyValue)) {
              validationSkipped = true;
              validationSkipReason = "unsupported key type";
              break;
            }
          bool valueValid = true;
          if (requestHostData[0].spec.nullCount > 0 &&
              !isValidAt(requestHostData[0], row)) {
            valueValid = false;
          }
          if (valueValid) {
            __int128_t value{};
            if (!decodeValueInt128(requestHostData[0], row, value)) {
              validationSkipped = true;
              validationSkipReason = "unsupported request type";
              break;
            }
            expectedSum[keyValue] += value;
            expectedCount[keyValue] += 1;
          }
          expectedCountAll[keyValue] += 1;
          }

          if (!validationSkipped) {
            if (!outputKeys || outputResults.empty()) {
              validationSkipped = true;
              validationSkipReason = "missing output for validation";
            } else if (outputKeys->num_columns() != 1 ||
                     outputResults[0].results.size() != requestAggKinds[0].size()) {
              validationSkipped = true;
              validationSkipReason = "output shape mismatch";
            } else {
              auto outputKeyHost =
                  readOutputColumnHostData(outputKeys->view().column(0), stream);
            if (outputKeyHost.spec.nullCount > 0 ||
                outputResults[0].results.empty()) {
                validationSkipped = true;
              validationSkipReason = "output contains nulls or is empty";
              } else {
              std::vector<ColumnHostData> outputValHosts;
              outputValHosts.reserve(outputResults[0].results.size());
              for (auto const& col : outputResults[0].results) {
                auto host = readOutputColumnHostData(col->view(), stream);
                if (host.spec.nullCount > 0) {
                  validationSkipped = true;
                  validationSkipReason = "output contains nulls";
                  break;
                }
                outputValHosts.push_back(std::move(host));
              }
              if (validationSkipped) {
                // do not proceed
              } else {
                auto const outRows = outputKeyHost.spec.size;
                for (int64_t row = 0; row < outRows; ++row) {
                  __int128_t keyValue{};
                  if (!decodeValueInt128(outputKeyHost, row, keyValue)) {
                    validationSkipped = true;
                    validationSkipReason = "unsupported output key type";
                    break;
                  }
                auto sumIt = expectedSum.find(keyValue);
                auto countIt = expectedCount.find(keyValue);
                auto countAllIt = expectedCountAll.find(keyValue);
                for (size_t aggIdx = 0; aggIdx < requestAggKinds[0].size();
                     ++aggIdx) {
                  auto kind = requestAggKinds[0][aggIdx];
                  __int128_t outValue{};
                  if (!decodeValueInt128(outputValHosts[aggIdx], row, outValue)) {
                    validationSkipped = true;
                    validationSkipReason = "unsupported output value type";
                    break;
                  }
                  __int128_t expectedValue = 0;
                  if (kind == cudf::aggregation::Kind::SUM) {
                    expectedValue = (sumIt != expectedSum.end()) ? sumIt->second : 0;
                  } else if (kind == cudf::aggregation::Kind::COUNT_VALID) {
                    expectedValue =
                        (countIt != expectedCount.end()) ? countIt->second : 0;
                  } else if (kind == cudf::aggregation::Kind::COUNT_ALL) {
                    expectedValue =
                        (countAllIt != expectedCountAll.end()) ? countAllIt->second
                                                               : 0;
                  }
                  if (outValue != expectedValue) {
                    if (validationMismatches == 0) {
                      firstMismatchKey = keyValue;
                      firstExpected = expectedValue;
                      firstActual = outValue;
                    }
                    ++validationMismatches;
                  }
                }
                if (validationSkipped) {
                  break;
                }
                }
                if (!validationSkipped &&
                  static_cast<int64_t>(expectedSum.size()) != outRows) {
                auto diff = static_cast<int64_t>(expectedSum.size()) - outRows;
                  validationMismatches += diff >= 0 ? diff : -diff;
                }
              }
              }
            }
          }
        }
      }

      if (validate) {
        if (validationSkipped) {
          std::cout << "[HashAggDumpReplay] validate=skipped reason="
                    << validationSkipReason << std::endl;
        } else {
          std::cout << "[HashAggDumpReplay] validate=mismatches "
                    << validationMismatches;
          if (validationMismatches > 0) {
            std::cout << " firstMismatchKey=" << toString128(firstMismatchKey)
                      << " expected=" << toString128(firstExpected)
                      << " actual=" << toString128(firstActual);
          }
          std::cout << std::endl;
        }
      }

      auto const failed = aggregateThrew || peekErr != cudaSuccess ||
          streamSyncErr != cudaSuccess || deviceSyncErr != cudaSuccess ||
          (!validationSkipped && validationMismatches > 0);
      return failed ? 2 : 0;
    };

    if (std::filesystem::is_directory(inputPath) &&
        std::filesystem::exists(inputPath / "hashagg_dump_index.txt")) {
      if (replayOptions.replayChain) {
        return runChainSession(inputPath, validateOptions);
      }
      auto dumps = readSessionIndex(inputPath);
      int failedCount = 0;
      for (auto const& dumpDir : dumps) {
        auto code = runOne(dumpDir / "manifest.txt");
        if (code != 0) {
          ++failedCount;
        }
      }
      std::cout << "[HashAggDumpReplay] session_dumps=" << dumps.size()
                << " failures=" << failedCount << std::endl;
      return failedCount > 0 ? 2 : 0;
    }

    return runOne(inputPath);
  } catch (const std::exception& e) {
    std::cerr << "[HashAggDumpReplay] fatal error: " << e.what() << std::endl;
    return 1;
  }
}
