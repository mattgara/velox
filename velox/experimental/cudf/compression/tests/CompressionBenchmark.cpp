/* Temporary local-only throughput harness for PackedColumnsCodec. */
#include "velox/experimental/cudf/compression/PackedColumnsCodec.h"

#include <cuda_runtime_api.h>
#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::compression {
namespace {

using Clock = std::chrono::steady_clock;

struct Case {
  std::string_view name;
  CompressionOptions options;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                : values[middle];
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

std::vector<int64_t> makeValues(std::size_t rows, bool monotonic) {
  std::vector<int64_t> values(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    if (monotonic) {
      values[row] = 1'700'000'000'000LL + static_cast<int64_t>(row) * 1'000LL;
      continue;
    }
    auto mixed = static_cast<uint64_t>(row) + 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    mixed ^= mixed >> 31;
    values[row] =
        5'000'000'000LL + static_cast<int64_t>(mixed & ((1u << 20) - 1));
  }
  return values;
}

std::unique_ptr<cudf::column> makeColumn(
    const std::vector<int64_t>& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  rmm::device_buffer data{
      values.data(), values.size() * sizeof(int64_t), stream, memoryResource};
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT64},
      static_cast<cudf::size_type>(values.size()),
      std::move(data),
      rmm::device_buffer{},
      0);
}

void verifyEqual(
    const rmm::device_buffer& expected,
    const rmm::device_buffer& actual,
    rmm::cuda_stream_view stream) {
  if (expected.size() != actual.size()) {
    throw std::runtime_error("decoded allocation has the wrong size");
  }
  constexpr std::size_t kChunkBytes = 16u << 20;
  std::vector<uint8_t> expectedHost(std::min(kChunkBytes, expected.size()));
  std::vector<uint8_t> actualHost(expectedHost.size());
  for (std::size_t offset = 0; offset < expected.size();
       offset += kChunkBytes) {
    const auto bytes = std::min(kChunkBytes, expected.size() - offset);
    const auto* expectedData =
        static_cast<const uint8_t*>(expected.data()) + offset;
    const auto* actualData =
        static_cast<const uint8_t*>(actual.data()) + offset;
    if (cudaMemcpyAsync(
            expectedHost.data(),
            expectedData,
            bytes,
            cudaMemcpyDeviceToHost,
            stream.value()) != cudaSuccess ||
        cudaMemcpyAsync(
            actualHost.data(),
            actualData,
            bytes,
            cudaMemcpyDeviceToHost,
            stream.value()) != cudaSuccess) {
      throw std::runtime_error("correctness copy failed");
    }
    stream.synchronize();
    if (!std::equal(
            expectedHost.begin(),
            expectedHost.begin() + bytes,
            actualHost.begin())) {
      throw std::runtime_error("decoded allocation differs from input");
    }
  }
}

template <typename Function>
std::vector<double> measure(int warmups, int iterations, Function&& function) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    function();
  }
  std::vector<double> seconds;
  seconds.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    seconds.push_back(std::chrono::duration<double>(stop - start).count());
  }
  return seconds;
}

void runCase(
    const cudf::packed_columns& packed,
    std::string_view dataset,
    const Case& benchmarkCase,
    int warmups,
    int iterations,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  PackedColumnsCodec codec{stream, memoryResource, memoryResource};
  auto reference = codec.compress(packed, benchmarkCase.options);
  if (!reference) {
    std::cout << "RESULT dataset=" << dataset << " case=" << benchmarkCase.name
              << " accepted=false\n";
    return;
  }
  auto decoded = codec.decompress(
      {static_cast<const uint8_t*>(reference->data.data()),
       reference->data.size()},
      reference->descriptor);
  verifyEqual(*packed.gpu_data, decoded, stream);

  const auto encodeSeconds = measure(warmups, iterations, [&] {
    auto compressed = codec.compress(packed, benchmarkCase.options);
    if (!compressed) {
      throw std::runtime_error("compression was inconsistently rejected");
    }
  });
  const auto decodeSeconds = measure(warmups, iterations, [&] {
    auto output = codec.decompress(
        {static_cast<const uint8_t*>(reference->data.data()),
         reference->data.size()},
        reference->descriptor);
    if (output.size() != packed.gpu_data->size()) {
      throw std::runtime_error("decompression returned the wrong size");
    }
  });

  const auto inputBytes = static_cast<double>(packed.gpu_data->size());
  const auto encodeMedian = median(encodeSeconds);
  const auto decodeMedian = median(decodeSeconds);
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT dataset=" << dataset << " case=" << benchmarkCase.name
            << " accepted=true input_bytes=" << packed.gpu_data->size()
            << " encoded_bytes=" << reference->data.size() << " ratio="
            << static_cast<double>(reference->data.size()) / inputBytes
            << " encode_ms=" << encodeMedian * 1e3
            << " encode_gbps=" << inputBytes / encodeMedian / 1e9
            << " encode_p10_ms=" << percentile(encodeSeconds, 0.10) * 1e3
            << " encode_p90_ms=" << percentile(encodeSeconds, 0.90) * 1e3
            << " decode_ms=" << decodeMedian * 1e3
            << " decode_gbps=" << inputBytes / decodeMedian / 1e9
            << " decode_p10_ms=" << percentile(decodeSeconds, 0.10) * 1e3
            << " decode_p90_ms=" << percentile(decodeSeconds, 0.90) * 1e3
            << '\n';
}

} // namespace
} // namespace facebook::velox::cudf_velox::compression

int main(int argc, char** argv) {
  using namespace facebook::velox::cudf_velox::compression;
  const std::size_t targetMiB =
      argc > 1 ? std::stoull(argv[1]) : std::size_t{256};
  const int iterations = argc > 2 ? std::stoi(argv[2]) : 9;
  const int warmups = argc > 3 ? std::stoi(argv[3]) : 2;
  const std::string_view caseFilter = argc > 4 ? argv[4] : "";
  const std::string_view datasetFilter = argc > 5 ? argv[5] : "";
  const auto targetBytes = targetMiB << 20;
  const auto rows = targetBytes / sizeof(int64_t);

  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    throw std::runtime_error("failed to query CUDA device");
  }
  std::cout << "BENCHMARK gpu=" << device << " name=\"" << properties.name
            << "\" target_mib=" << targetMiB << " rows=" << rows
            << " warmups=" << warmups << " iterations=" << iterations << '\n';

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource asyncMemoryResource;
  const auto memoryResource =
      rmm::device_async_resource_ref{asyncMemoryResource};
  const std::vector<Case> cases{
      {"automatic_ans", {}},
      {"for_ans", {NumericTransform::kFrameOfReference, EntropyEncoding::kAns}},
      {"for_none",
       {NumericTransform::kFrameOfReference, EntropyEncoding::kNone}},
      {"delta_ans",
       {NumericTransform::kDeltaFrameOfReference, EntropyEncoding::kAns}},
      {"delta_none",
       {NumericTransform::kDeltaFrameOfReference, EntropyEncoding::kNone}},
      {"simpatico_auto",
       {NumericTransform::kAutomatic,
        EntropyEncoding::kAns,
        TypedRegionCodec::kSimpaticoBitpack}},
      {"simpatico_for",
       {NumericTransform::kFrameOfReference,
        EntropyEncoding::kAns,
        TypedRegionCodec::kSimpaticoBitpack}},
      {"simpatico_delta",
       {NumericTransform::kDeltaFrameOfReference,
        EntropyEncoding::kAns,
        TypedRegionCodec::kSimpaticoBitpack}},
  };

  for (const bool monotonic : {false, true}) {
    const std::string_view dataset = monotonic ? "monotonic" : "bounded";
    if (!datasetFilter.empty() && dataset != datasetFilter) {
      continue;
    }
    auto values = makeValues(rows, monotonic);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeColumn(values, stream.view(), memoryResource));
    cudf::table table{std::move(columns)};
    auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
    stream.synchronize();
    for (const auto& benchmarkCase : cases) {
      if (!caseFilter.empty() && benchmarkCase.name != caseFilter) {
        continue;
      }
      runCase(
          packed,
          dataset,
          benchmarkCase,
          warmups,
          iterations,
          stream.view(),
          memoryResource);
    }
  }
  return EXIT_SUCCESS;
}
