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
#include "velox/experimental/ucx-exchange/UcxFloat64AlpCodec.h"

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

void checkAlpCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void expectAlpBitExactRoundTrip(const std::vector<uint64_t>& inputBits) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputBits.data(), inputBits.size() * sizeof(uint64_t), stream.view());
  auto compressed = compressFloat64Alp(
      reinterpret_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputBits.size()),
      stream.view());
  auto output = decompressFloat64Alp(compressed, stream.view());

  std::vector<uint64_t> outputBits(inputBits.size());
  checkAlpCuda(cudaMemcpy(
      outputBits.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(outputBits, inputBits);
}

TEST(UcxFloat64AlpCodecTest, preservesEveryBitIncludingExceptions) {
  std::vector<uint64_t> bits{
      0x0000000000000000ULL,
      0x8000000000000000ULL,
      0x3ff0000000000000ULL,
      0xbff0000000000000ULL,
      0x0010000000000000ULL,
      0x0000000000000001ULL,
      0x7fefffffffffffffULL,
      0xffefffffffffffffULL,
      0x7ff0000000000000ULL,
      0xfff0000000000000ULL,
      0x7ff8000000000042ULL,
      0xfff0000000000042ULL,
  };
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (uint32_t index = 0; index < 65'537; ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    bits.push_back(state);
  }
  expectAlpBitExactRoundTrip(bits);
}

TEST(UcxFloat64AlpCodecTest, compressesTpchLikeDecimals) {
  constexpr uint32_t kValues = 1u << 20;
  std::vector<double> values(kValues);
  for (uint32_t index = 0; index < kValues; ++index) {
    values[index] =
        static_cast<double>((index * 48'271ULL) % 10'000'000ULL) / 100.0;
  }

  rmm::cuda_stream stream;
  rmm::device_buffer input(
      values.data(), values.size() * sizeof(double), stream.view());
  auto compressed = compressFloat64Alp(
      static_cast<const double*>(input.data()), kValues, stream.view());
  EXPECT_TRUE(compressed.used);
  EXPECT_LE(compressed.exponentIndex, 18);
  EXPECT_LE(compressed.factorIndex, compressed.exponentIndex);
  EXPECT_LT(compressed.candidateBytes, compressed.inputBytes / 2);
  auto output = decompressFloat64Alp(compressed, stream.view());
  std::vector<double> decoded(kValues);
  checkAlpCuda(cudaMemcpy(
      decoded.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(0, std::memcmp(decoded.data(), values.data(), output.size()));
}

struct AlpMeasurement {
  bool used{false};
  uint32_t exponent{0};
  uint32_t factor{0};
  uint32_t bitWidth{0};
  uint32_t exceptions{0};
  std::size_t inputBytes{0};
  std::size_t candidateBytes{0};
  double encodeSeconds{0};
  double decodeSeconds{0};
};

double alpMedian(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

AlpMeasurement measureAlp(const std::vector<double>& inputValues) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(
      inputValues.data(), inputValues.size() * sizeof(double), stream.view());
  checkAlpCuda(cudaStreamSynchronize(stream.value()));

  auto warm = compressFloat64Alp(
      static_cast<const double*>(input.data()),
      static_cast<uint32_t>(inputValues.size()),
      stream.view());
  decompressFloat64Alp(warm, stream.view());

  constexpr uint32_t kIterations = 5;
  std::vector<double> encodeTimes;
  Float64AlpCompressResult compressed;
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    auto current = compressFloat64Alp(
        static_cast<const double*>(input.data()),
        static_cast<uint32_t>(inputValues.size()),
        stream.view());
    encodeTimes.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    compressed = std::move(current);
  }

  std::vector<double> decodeTimes;
  rmm::device_buffer output;
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    auto current = decompressFloat64Alp(compressed, stream.view());
    decodeTimes.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    output = std::move(current);
  }

  std::vector<double> decoded(inputValues.size());
  checkAlpCuda(cudaMemcpy(
      decoded.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(0, std::memcmp(decoded.data(), inputValues.data(), output.size()));

  return AlpMeasurement{
      compressed.used,
      compressed.exponentIndex,
      compressed.factorIndex,
      compressed.bitWidth,
      compressed.exceptionCount,
      compressed.inputBytes,
      compressed.candidateBytes,
      alpMedian(std::move(encodeTimes)),
      alpMedian(std::move(decodeTimes))};
}

void logAlpMeasurement(
    const std::string& name,
    const AlpMeasurement& measurement) {
  constexpr double kLinkBytesPerSecond = 25e9;
  const double rawWireSeconds = measurement.inputBytes / kLinkBytesPerSecond;
  const double compressedTotalSeconds = measurement.encodeSeconds +
      measurement.candidateBytes / kLinkBytesPerSecond +
      measurement.decodeSeconds;
  const double byteReduction = 1.0 -
      static_cast<double>(measurement.candidateBytes) / measurement.inputBytes;
  const double modeledReduction = 1.0 - compressedTotalSeconds / rawWireSeconds;

  LOG(INFO) << "G-ALP FP64 " << name << " used=" << measurement.used
            << " exponent=" << measurement.exponent
            << " factor=" << measurement.factor
            << " bitWidth=" << measurement.bitWidth
            << " exceptions=" << measurement.exceptions << " rawMiB="
            << measurement.inputBytes / static_cast<double>(1 << 20)
            << " candidateMiB="
            << measurement.candidateBytes / static_cast<double>(1 << 20)
            << " byteReduction=" << byteReduction * 100.0 << "% enc="
            << measurement.inputBytes / measurement.encodeSeconds / 1e9
            << " GB/s dec="
            << measurement.inputBytes / measurement.decodeSeconds / 1e9
            << " GB/s modeled25GB/sReduction=" << modeledReduction * 100.0
            << "%";
}

TEST(UcxFloat64AlpCodecTest, DISABLED_tpchLikeBenchmark25Gbps) {
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
    const auto result = measureAlp(values);
    EXPECT_TRUE(result.used);
    logAlpMeasurement(name, result);
  }
}

} // namespace
} // namespace facebook::velox::ucx_exchange
