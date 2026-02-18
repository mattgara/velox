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
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
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

struct ReplayOptions {
  std::filesystem::path manifestPath;
  bool dumpInputSamples{false};
  bool showHelp{false};
};

struct ColumnBuffers {
  std::vector<uint8_t> data;
  std::vector<uint8_t> mask;
};

constexpr int32_t kSampleCount = 10;

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

std::string usageString() {
  std::ostringstream out;
  out << "usage: velox_cudf_hashagg_replay --manifest <path/to/manifest.txt> "
      << "[--dump_input_samples]\n"
      << "   or: velox_cudf_hashagg_replay --dump_dir <path/to/dump_dir> "
      << "[--dump_input_samples]\n"
      << "   --dump_input_samples prints samples and exits";
  return out.str();
}

ReplayOptions parseOptions(int argc, char** argv) {
  ReplayOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if ((arg == "-h") || (arg == "--help")) {
      options.showHelp = true;
      continue;
    }
    if (arg == "--manifest" && i + 1 < argc) {
      options.manifestPath = argv[++i];
      continue;
    }
    if (arg == "--dump_dir" && i + 1 < argc) {
      options.manifestPath = std::filesystem::path(argv[++i]) / "manifest.txt";
      continue;
    }
    if (arg == "--dump_input_samples" || arg == "--dump-input-samples") {
      options.dumpInputSamples = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg + "\n" + usageString());
  }
  if (!options.showHelp && options.manifestPath.empty()) {
    throw std::runtime_error(usageString());
  }
  return options;
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

std::string typeIdToString(cudf::type_id typeId) {
  switch (typeId) {
    case cudf::type_id::EMPTY:
      return "EMPTY";
    case cudf::type_id::INT8:
      return "INT8";
    case cudf::type_id::INT16:
      return "INT16";
    case cudf::type_id::INT32:
      return "INT32";
    case cudf::type_id::INT64:
      return "INT64";
    case cudf::type_id::UINT8:
      return "UINT8";
    case cudf::type_id::UINT16:
      return "UINT16";
    case cudf::type_id::UINT32:
      return "UINT32";
    case cudf::type_id::UINT64:
      return "UINT64";
    case cudf::type_id::FLOAT32:
      return "FLOAT32";
    case cudf::type_id::FLOAT64:
      return "FLOAT64";
    case cudf::type_id::BOOL8:
      return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS:
      return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS:
      return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS:
      return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS:
      return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS:
      return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS:
      return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS:
      return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32:
      return "DICTIONARY32";
    case cudf::type_id::STRING:
      return "STRING";
    case cudf::type_id::LIST:
      return "LIST";
    case cudf::type_id::DECIMAL32:
      return "DECIMAL32";
    case cudf::type_id::DECIMAL64:
      return "DECIMAL64";
    case cudf::type_id::DECIMAL128:
      return "DECIMAL128";
    case cudf::type_id::STRUCT:
      return "STRUCT";
    default:
      break;
  }
  return "UNKNOWN";
}

std::string int128ToString(__int128 value) {
  if (value == 0) {
    return "0";
  }
  bool const negative = value < 0;
  unsigned __int128 u = 0;
  if (negative) {
    u = static_cast<unsigned __int128>(-(value + 1));
    u += 1;
  } else {
    u = static_cast<unsigned __int128>(value);
  }
  std::string digits;
  while (u > 0) {
    auto digit = static_cast<int>(u % 10);
    digits.push_back(static_cast<char>('0' + digit));
    u /= 10;
  }
  if (negative) {
    digits.push_back('-');
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

template <typename IntT>
std::string formatScaledInteger(IntT value, int32_t scale) {
  std::string digits;
  if constexpr (std::is_same_v<IntT, __int128>) {
    digits = int128ToString(value);
  } else {
    digits = std::to_string(value);
  }
  bool negative = false;
  if (!digits.empty() && digits[0] == '-') {
    negative = true;
    digits = digits.substr(1);
  }
  if (scale >= 0) {
    digits.append(static_cast<size_t>(scale), '0');
  } else {
    auto const fracDigits = static_cast<size_t>(-scale);
    if (digits.size() <= fracDigits) {
      auto const zeros = fracDigits - digits.size();
      digits = "0." + std::string(zeros, '0') + digits;
    } else {
      digits.insert(digits.end() - static_cast<std::ptrdiff_t>(fracDigits), '.');
    }
  }
  return negative ? "-" + digits : digits;
}

std::string bytesToHex(uint8_t const* data, int64_t size) {
  std::ostringstream out;
  out << "0x";
  for (int64_t i = 0; i < size; ++i) {
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(data[i]);
  }
  return out.str();
}

bool isValidAt(std::vector<uint8_t> const& mask, int64_t index) {
  if (mask.empty()) {
    return true;
  }
  auto const byteIndex = static_cast<size_t>(index / 8);
  auto const bitIndex = static_cast<uint8_t>(index % 8);
  if (byteIndex >= mask.size()) {
    return false;
  }
  return (mask[byteIndex] & static_cast<uint8_t>(1u << bitIndex)) != 0;
}

std::string formatElementValue(
    ColumnSpec const& spec,
    uint8_t const* data) {
  auto const typeId = static_cast<cudf::type_id>(spec.typeId);
  switch (typeId) {
    case cudf::type_id::INT8: {
      int8_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::INT16: {
      int16_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::INT32: {
      int32_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::INT64: {
      int64_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::UINT8: {
      uint8_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::UINT16: {
      uint16_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::UINT32: {
      uint32_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::UINT64: {
      uint64_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::FLOAT32: {
      float value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      std::ostringstream out;
      out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
      return out.str();
    }
    case cudf::type_id::FLOAT64: {
      double value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      std::ostringstream out;
      out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
      return out.str();
    }
    case cudf::type_id::BOOL8: {
      uint8_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return value ? "true" : "false";
    }
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS: {
      int32_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS: {
      int64_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return std::to_string(value);
    }
    case cudf::type_id::DECIMAL32: {
      int32_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return formatScaledInteger(value, spec.scale) + " (raw=" +
          std::to_string(value) + ", scale=" + std::to_string(spec.scale) + ")";
    }
    case cudf::type_id::DECIMAL64: {
      int64_t value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return formatScaledInteger(value, spec.scale) + " (raw=" +
          std::to_string(value) + ", scale=" + std::to_string(spec.scale) + ")";
    }
    case cudf::type_id::DECIMAL128: {
      __int128 value;
      if (spec.elementSize != static_cast<int64_t>(sizeof(value))) {
        return bytesToHex(data, spec.elementSize);
      }
      std::memcpy(&value, data, sizeof(value));
      return formatScaledInteger(value, spec.scale) + " (raw=" +
          int128ToString(value) + ", scale=" + std::to_string(spec.scale) + ")";
    }
    default:
      break;
  }
  return bytesToHex(data, spec.elementSize);
}

std::string formatSampleList(
    ColumnSpec const& spec,
    std::vector<uint8_t> const& data,
    std::vector<uint8_t> const& mask,
    std::vector<int64_t> const& indices) {
  std::ostringstream out;
  for (size_t i = 0; i < indices.size(); ++i) {
    auto const index = indices[i];
    if (i > 0) {
      out << ", ";
    }
    out << index << "=";
    if (!isValidAt(mask, index)) {
      out << "NULL";
      continue;
    }
    auto const offset = static_cast<size_t>(index) * static_cast<size_t>(spec.elementSize);
    auto const* element = data.data() + offset;
    out << formatElementValue(spec, element);
  }
  return out.str();
}

std::vector<int64_t> firstIndices(int64_t size) {
  auto const count = std::min<int64_t>(kSampleCount, size);
  std::vector<int64_t> indices;
  indices.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    indices.push_back(i);
  }
  return indices;
}

std::vector<int64_t> lastIndices(int64_t size) {
  auto const count = std::min<int64_t>(kSampleCount, size);
  std::vector<int64_t> indices;
  indices.reserve(static_cast<size_t>(count));
  auto const start = std::max<int64_t>(0, size - count);
  for (int64_t i = start; i < size; ++i) {
    indices.push_back(i);
  }
  return indices;
}

std::vector<int64_t> randomIndices(
    int64_t size,
    std::mt19937& rng) {
  auto const count = std::min<int64_t>(kSampleCount, size);
  if (count <= 0) {
    return {};
  }
  std::unordered_set<int64_t> selected;
  std::uniform_int_distribution<int64_t> dist(0, size - 1);
  while (static_cast<int64_t>(selected.size()) < count) {
    selected.insert(dist(rng));
  }
  std::vector<int64_t> indices(selected.begin(), selected.end());
  std::sort(indices.begin(), indices.end());
  return indices;
}

void dumpInputSamples(
    std::string const& label,
    ColumnSpec const& spec,
    std::vector<uint8_t> const& data,
    std::vector<uint8_t> const& mask,
    std::mt19937& rng) {
  auto const typeId = static_cast<cudf::type_id>(spec.typeId);
  std::cout << "[HashAggReplay] input=" << label
            << " type=" << typeIdToString(typeId)
            << " size=" << spec.size
            << " elementSize=" << spec.elementSize
            << " nullCount=" << spec.nullCount
            << " scale=" << spec.scale
            << " dataFile=" << spec.dataFile;
  if (!spec.nullMaskFile.empty()) {
    std::cout << " nullMaskFile=" << spec.nullMaskFile;
  }
  std::cout << std::endl;
  auto const first = firstIndices(spec.size);
  auto const last = lastIndices(spec.size);
  auto const random = randomIndices(spec.size, rng);
  std::cout << "[HashAggReplay] input=" << label << " first="
            << formatSampleList(spec, data, mask, first) << std::endl;
  std::cout << "[HashAggReplay] input=" << label << " last="
            << formatSampleList(spec, data, mask, last) << std::endl;
  std::cout << "[HashAggReplay] input=" << label << " random="
            << formatSampleList(spec, data, mask, random) << std::endl;
}

ColumnBuffers readColumnBuffers(
    std::filesystem::path const& dumpDir,
    ColumnSpec const& spec) {
  ColumnBuffers buffers;
  buffers.data = readBinaryFile(dumpDir / spec.dataFile);
  auto const expectedBytes =
      static_cast<size_t>(spec.size) * static_cast<size_t>(spec.elementSize);
  if (buffers.data.size() != expectedBytes) {
    throw std::runtime_error(
        "data size mismatch for " + spec.dataFile + ": expected " +
        std::to_string(expectedBytes) + " got " + std::to_string(buffers.data.size()));
  }

  if (spec.nullCount > 0) {
    if (spec.nullMaskFile.empty()) {
      throw std::runtime_error(
          "null_count > 0 but null mask file missing for " + spec.dataFile);
    }
    buffers.mask = readBinaryFile(dumpDir / spec.nullMaskFile);
    auto const expectedMaskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(
            static_cast<cudf::size_type>(spec.size)));
    if (buffers.mask.size() != expectedMaskBytes) {
      throw std::runtime_error(
          "null-mask size mismatch for " + spec.nullMaskFile + ": expected " +
          std::to_string(expectedMaskBytes) + " got " +
          std::to_string(buffers.mask.size()));
    }
  }
  return buffers;
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

std::unique_ptr<cudf::column> makeColumnFromHost(
    std::filesystem::path const& dumpDir,
    ColumnSpec const& spec,
    rmm::cuda_stream_view stream,
    std::string const& label,
    bool dumpSamples,
    std::mt19937& rng) {
  auto const size = static_cast<cudf::size_type>(spec.size);
  auto const type = cudf::data_type{
      static_cast<cudf::type_id>(spec.typeId), spec.scale};
  auto col = cudf::make_fixed_width_column(
      type, size, cudf::mask_state::UNALLOCATED, stream);

  auto buffers = readColumnBuffers(dumpDir, spec);

  if (dumpSamples) {
    dumpInputSamples(label, spec, buffers.data, buffers.mask, rng);
  }

  auto const expectedBytes =
      static_cast<size_t>(spec.size) * static_cast<size_t>(spec.elementSize);
  if (expectedBytes > 0) {
    auto const copyStatus = cudaMemcpyAsync(
        col->mutable_view().data<uint8_t>(),
        buffers.data.data(),
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

  if (spec.nullCount > 0) {
    auto const expectedMaskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(size));
    rmm::device_buffer maskBuffer(expectedMaskBytes, stream);
    if (expectedMaskBytes > 0) {
      auto const copyStatus = cudaMemcpyAsync(
          maskBuffer.data(),
          buffers.mask.data(),
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
    col->set_null_mask(std::move(maskBuffer), static_cast<cudf::size_type>(spec.nullCount));
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

} // namespace

int main(int argc, char** argv) {
  try {
    auto options = parseOptions(argc, argv);
    if (options.showHelp) {
      std::cout << usageString() << std::endl;
      return 0;
    }
    auto const manifestPath = options.manifestPath;
    auto const dumpDir = manifestPath.parent_path();
    auto const manifest = parseManifest(manifestPath);
    auto const stream = cudf::get_default_stream();
    std::mt19937 rng(0x5a5a5a5a);

    auto const keyCount = requireInt32(manifest, "key_count");
    auto const requestCount = requireInt32(manifest, "request_count");
    auto const nullPolicyValue = requireInt32(manifest, "null_policy");
    auto const nullPolicy = static_cast<cudf::null_policy>(nullPolicyValue);

    if (options.dumpInputSamples) {
      for (int32_t i = 0; i < keyCount; ++i) {
        auto spec = parseColumnSpec(manifest, "key." + std::to_string(i));
        auto buffers = readColumnBuffers(dumpDir, spec);
        dumpInputSamples(
            "key." + std::to_string(i), spec, buffers.data, buffers.mask, rng);
      }
      for (int32_t i = 0; i < requestCount; ++i) {
        auto const prefix = "request." + std::to_string(i);
        auto spec = parseColumnSpec(manifest, prefix);
        auto buffers = readColumnBuffers(dumpDir, spec);
        dumpInputSamples(
            prefix, spec, buffers.data, buffers.mask, rng);
      }
      return 0;
    }

    std::vector<std::unique_ptr<cudf::column>> keyColumns;
    std::vector<cudf::column_view> keyViews;
    keyColumns.reserve(keyCount);
    keyViews.reserve(keyCount);
    for (int32_t i = 0; i < keyCount; ++i) {
      auto spec = parseColumnSpec(manifest, "key." + std::to_string(i));
      auto col = makeColumnFromHost(
          dumpDir, spec, stream, "key." + std::to_string(i),
          options.dumpInputSamples, rng);
      keyViews.push_back(col->view());
      keyColumns.push_back(std::move(col));
    }
    cudf::table_view groupbyKeys(keyViews);

    std::vector<std::unique_ptr<cudf::column>> requestValueColumns;
    std::vector<cudf::groupby::aggregation_request> requests;
    requestValueColumns.reserve(requestCount);
    requests.reserve(requestCount);
    for (int32_t i = 0; i < requestCount; ++i) {
      auto const prefix = "request." + std::to_string(i);
      auto spec = parseColumnSpec(manifest, prefix);
      auto col = makeColumnFromHost(
          dumpDir, spec, stream, "request." + std::to_string(i),
          options.dumpInputSamples, rng);
      requestValueColumns.push_back(std::move(col));

      cudf::groupby::aggregation_request request;
      request.values = requestValueColumns.back()->view();
      auto const aggCount = requireInt32(manifest, prefix + ".aggregation_count");
      request.aggregations.reserve(aggCount);
      for (int32_t aggIdx = 0; aggIdx < aggCount; ++aggIdx) {
        auto const kindValue = requireInt32(
            manifest, prefix + ".aggregation_kind." + std::to_string(aggIdx));
        request.aggregations.push_back(makeGroupbyAggregation(
            static_cast<cudf::aggregation::Kind>(kindValue)));
      }
      requests.push_back(std::move(request));
    }

    std::cout << "[HashAggReplay] manifest=" << manifestPath.string()
              << " keyCount=" << keyCount << " requestCount=" << requestCount
              << " nullPolicy=" << nullPolicyValue << std::endl;

    bool aggregateThrew = false;
    std::string exceptionText;
    int64_t outputRows = -1;
    int64_t outputCount = -1;
    try {
      cudf::groupby::groupby groupby(groupbyKeys, nullPolicy);
      auto output = groupby.aggregate(requests, stream);
      outputRows = output.first ? output.first->num_rows() : 0;
      outputCount = output.second.size();
    } catch (const std::exception& e) {
      aggregateThrew = true;
      exceptionText = e.what();
    }

    auto const peekErr = cudaPeekAtLastError();
    auto const streamSyncErr = cudaStreamSynchronize(stream.value());
    auto const deviceSyncErr = cudaDeviceSynchronize();

    std::cout << "[HashAggReplay] aggregateThrew=" << (aggregateThrew ? 1 : 0)
              << " outputRows=" << outputRows
              << " outputCount=" << outputCount
              << " cudaPeek=" << cudaGetErrorName(peekErr) << "("
              << static_cast<int>(peekErr) << ")"
              << " cudaStreamSync=" << cudaGetErrorName(streamSyncErr) << "("
              << static_cast<int>(streamSyncErr) << ")"
              << " cudaDeviceSync=" << cudaGetErrorName(deviceSyncErr) << "("
              << static_cast<int>(deviceSyncErr) << ")" << std::endl;
    if (aggregateThrew) {
      std::cout << "[HashAggReplay] exception=" << exceptionText << std::endl;
    }

    auto const failed = aggregateThrew || peekErr != cudaSuccess ||
        streamSyncErr != cudaSuccess || deviceSyncErr != cudaSuccess;
    return failed ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "[HashAggReplay] fatal error: " << e.what() << std::endl;
    return 1;
  }
}
