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
#include "velox/experimental/ucx-exchange/UcxFloat64Codec.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

namespace facebook::velox::ucx_exchange {
namespace {

void checkCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void expectBitExactRoundTrip(
    const std::vector<uint64_t>& inputBits,
    uint32_t exponentPlanes) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputBits.data(), inputBits.size() * sizeof(uint64_t), stream.view());
  auto compressed = compressFloat64(
      reinterpret_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputBits.size()),
      exponentPlanes,
      stream.view());
  auto output = decompressFloat64(compressed, stream.view());

  std::vector<uint64_t> outputBits(inputBits.size());
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(
          outputBits.data(),
          output.data(),
          output.size(),
          cudaMemcpyDeviceToHost));
  EXPECT_EQ(outputBits, inputBits);
}

TEST(UcxFloat64CodecTest, preservesEveryBitWithOneOrTwoPlanes) {
  std::vector<uint64_t> bits{
      0x0000000000000000ULL, // +0
      0x8000000000000000ULL, // -0
      0x3ff0000000000000ULL, // +1
      0xbff0000000000000ULL, // -1
      0x0010000000000000ULL, // minimum normal
      0x0000000000000001ULL, // minimum subnormal
      0x7fefffffffffffffULL, // maximum finite
      0xffefffffffffffffULL, // minimum finite
      0x7ff0000000000000ULL, // +infinity
      0xfff0000000000000ULL, // -infinity
      0x7ff8000000000042ULL, // quiet NaN with payload
      0xfff0000000000042ULL, // negative signaling NaN payload
  };

  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (uint32_t index = 0; index < 65'537; ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    bits.push_back(state);
  }

  expectBitExactRoundTrip(bits, 1);
  expectBitExactRoundTrip(bits, 2);
}

TEST(UcxFloat64CodecTest, validatesPlaneCount) {
  rmm::cuda_stream stream;
  EXPECT_THROW(
      compressFloat64(nullptr, 0, 0, stream.view()), std::invalid_argument);
  EXPECT_THROW(
      compressFloat64(nullptr, 0, 3, stream.view()), std::invalid_argument);
}

struct Measurement {
  std::size_t inputBytes{0};
  std::size_t candidateBytes{0};
  double encodeSeconds{0};
  double decodeSeconds{0};
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

Measurement measure(
    const std::vector<double>& inputValues,
    uint32_t exponentPlanes) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputValues.data(), inputValues.size() * sizeof(double), stream.view());
  checkCuda(cudaStreamSynchronize(stream.value()));

  auto warm = compressFloat64(
      static_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputValues.size()),
      exponentPlanes,
      stream.view());
  decompressFloat64(warm, stream.view());

  constexpr uint32_t kIterations = 5;
  std::vector<double> encodeTimes;
  encodeTimes.reserve(kIterations);
  Float64CompressResult compressed;
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    auto current = compressFloat64(
        static_cast<const double*>(input.data()),
        static_cast<uint32_t>(inputValues.size()),
        exponentPlanes,
        stream.view());
    encodeTimes.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    compressed = std::move(current);
  }

  std::vector<double> decodeTimes;
  decodeTimes.reserve(kIterations);
  rmm::device_buffer output;
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    auto current = decompressFloat64(compressed, stream.view());
    decodeTimes.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    output = std::move(current);
  }

  std::vector<double> outputValues(inputValues.size());
  checkCuda(cudaMemcpy(
      outputValues.data(),
      output.data(),
      output.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      0, std::memcmp(outputValues.data(), inputValues.data(), output.size()));

  return Measurement{
      compressed.inputBytes,
      compressed.candidateBytes,
      median(std::move(encodeTimes)),
      median(std::move(decodeTimes))};
}

void logMeasurement(
    const std::string& name,
    uint32_t exponentPlanes,
    const Measurement& measurement) {
  constexpr double kLinkBytesPerSecond = 25e9;
  const double rawWireSeconds = measurement.inputBytes / kLinkBytesPerSecond;
  const double compressedWireSeconds =
      measurement.candidateBytes / kLinkBytesPerSecond;
  const double compressedTotalSeconds = measurement.encodeSeconds +
      compressedWireSeconds + measurement.decodeSeconds;
  const double reduction = 1.0 -
      static_cast<double>(measurement.candidateBytes) / measurement.inputBytes;
  const double modeledRuntimeReduction =
      1.0 - compressedTotalSeconds / rawWireSeconds;

  LOG(INFO) << "FP64 " << name << " planes=" << exponentPlanes << " rawMiB="
            << measurement.inputBytes / static_cast<double>(1 << 20)
            << " candidateMiB="
            << measurement.candidateBytes / static_cast<double>(1 << 20)
            << " byteReduction=" << reduction * 100.0 << "% enc="
            << measurement.inputBytes / measurement.encodeSeconds / 1e9
            << " GB/s dec="
            << measurement.inputBytes / measurement.decodeSeconds / 1e9
            << " GB/s modeled25GB/sReduction="
            << modeledRuntimeReduction * 100.0 << "%";
}

// This is a focused hardware experiment rather than a latency assertion. Run
// explicitly with --gtest_also_run_disabled_tests and its exact test filter.
TEST(UcxFloat64CodecTest, DISABLED_tpchLikeBenchmark25Gbps) {
  constexpr uint32_t kValues = 8u << 20;
  std::vector<double> values(kValues);

  const std::vector<std::pair<std::string, std::function<double(uint64_t)>>>
      data{
          {"extended_price",
           [](uint64_t index) {
             return static_cast<double>((index * 48'271ULL) % 10'000'000ULL) /
                 100.0;
           }},
          {"discount",
           [](uint64_t index) {
             return static_cast<double>((index * 13ULL) % 11ULL) / 100.0;
           }},
          {"account_balance",
           [](uint64_t index) {
             return static_cast<double>(
                        static_cast<int64_t>(
                            (index * 69'069ULL) % 20'000'000ULL) -
                        10'000'000LL) /
                 100.0;
           }},
      };

  for (const auto& [name, generate] : data) {
    for (uint32_t index = 0; index < kValues; ++index) {
      values[index] = generate(index);
    }
    for (uint32_t exponentPlanes = 1; exponentPlanes <= 2; ++exponentPlanes) {
      const auto result = measure(values, exponentPlanes);
      EXPECT_LT(result.candidateBytes, result.inputBytes);
      logMeasurement(name, exponentPlanes, result);
    }
  }
}

} // namespace
} // namespace facebook::velox::ucx_exchange
