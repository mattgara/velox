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
    rmm::cuda_stream_view stream) {
  auto const size = static_cast<cudf::size_type>(spec.size);
  auto const type = cudf::data_type{
      static_cast<cudf::type_id>(spec.typeId), spec.scale};
  auto col = cudf::make_fixed_width_column(
      type, size, cudf::mask_state::UNALLOCATED, stream);

  auto const hostData = readBinaryFile(dumpDir / spec.dataFile);
  auto const expectedBytes =
      static_cast<size_t>(spec.size) * static_cast<size_t>(spec.elementSize);
  if (hostData.size() != expectedBytes) {
    throw std::runtime_error(
        "data size mismatch for " + spec.dataFile + ": expected " +
        std::to_string(expectedBytes) + " got " + std::to_string(hostData.size()));
  }
  if (expectedBytes > 0) {
    auto const copyStatus = cudaMemcpyAsync(
        col->mutable_view().data<uint8_t>(),
        hostData.data(),
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
    if (spec.nullMaskFile.empty()) {
      throw std::runtime_error(
          "null_count > 0 but null mask file missing for " + spec.dataFile);
    }
    auto const hostMask = readBinaryFile(dumpDir / spec.nullMaskFile);
    auto const expectedMaskBytes =
        static_cast<size_t>(cudf::bitmask_allocation_size_bytes(size));
    if (hostMask.size() != expectedMaskBytes) {
      throw std::runtime_error(
          "null-mask size mismatch for " + spec.nullMaskFile + ": expected " +
          std::to_string(expectedMaskBytes) + " got " +
          std::to_string(hostMask.size()));
    }
    rmm::device_buffer maskBuffer(expectedMaskBytes, stream);
    if (expectedMaskBytes > 0) {
      auto const copyStatus = cudaMemcpyAsync(
          maskBuffer.data(),
          hostMask.data(),
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

std::filesystem::path parseManifestPath(int argc, char** argv) {
  std::filesystem::path manifestPath;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--manifest" && i + 1 < argc) {
      manifestPath = argv[++i];
      continue;
    }
    if (arg == "--dump_dir" && i + 1 < argc) {
      manifestPath = std::filesystem::path(argv[++i]) / "manifest.txt";
      continue;
    }
  }
  if (manifestPath.empty()) {
    throw std::runtime_error(
        "usage: velox_cudf_hashagg_replay --manifest <path/to/manifest.txt> "
        "or --dump_dir <path/to/dump_dir>");
  }
  return manifestPath;
}

} // namespace

int main(int argc, char** argv) {
  try {
    auto const manifestPath = parseManifestPath(argc, argv);
    auto const dumpDir = manifestPath.parent_path();
    auto const manifest = parseManifest(manifestPath);
    auto const stream = cudf::get_default_stream();

    auto const keyCount = requireInt32(manifest, "key_count");
    auto const requestCount = requireInt32(manifest, "request_count");
    auto const nullPolicyValue = requireInt32(manifest, "null_policy");
    auto const nullPolicy = static_cast<cudf::null_policy>(nullPolicyValue);

    std::vector<std::unique_ptr<cudf::column>> keyColumns;
    std::vector<cudf::column_view> keyViews;
    keyColumns.reserve(keyCount);
    keyViews.reserve(keyCount);
    for (int32_t i = 0; i < keyCount; ++i) {
      auto spec = parseColumnSpec(manifest, "key." + std::to_string(i));
      auto col = makeColumnFromHost(dumpDir, spec, stream);
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
      auto col = makeColumnFromHost(dumpDir, spec, stream);
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
