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
#include "velox/experimental/ucx-exchange/UcxCompression.h"

#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

namespace facebook::velox::ucx_exchange {
namespace {

// Uploads host bytes, round-trips through compress/decompress, checks
// byte-exactness, and reports {ratio, encode GB/s, decode GB/s}.
struct RoundTripResult {
  bool compressed;
  double ratio;
  double encodeGbps;
  double decodeGbps;
};

RoundTripResult roundTrip(const std::vector<uint8_t>& host) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(host.data(), host.size(), stream.view());
  cudaStreamSynchronize(stream.value());

  auto timedCompress = [&] {
    auto start = std::chrono::steady_clock::now();
    auto result = compressBlob(input.data(), input.size(), stream.view());
    cudaStreamSynchronize(stream.value());
    auto seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count();
    return std::make_pair(std::move(result), seconds);
  };
  // Warm-up then timed run (kernel JIT + pool warm-up dominate run one).
  timedCompress();
  auto [result, encodeSeconds] = timedCompress();

  RoundTripResult out{};
  out.compressed = result.used;
  if (!result.used) {
    return out;
  }
  std::size_t compressedBytes = 0;
  for (auto size : result.segSizes) {
    compressedBytes += size;
  }
  out.ratio = static_cast<double>(host.size()) / compressedBytes;
  out.encodeGbps = host.size() / encodeSeconds / 1e9;

  auto start = std::chrono::steady_clock::now();
  auto decoded = decompressBlob(
      result.data.data(), result.segSizes, host.size(), stream.view());
  cudaStreamSynchronize(stream.value());
  out.decodeGbps = host.size() /
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count() /
      1e9;

  std::vector<uint8_t> back(host.size());
  cudaMemcpy(back.data(), decoded.data(), back.size(), cudaMemcpyDeviceToHost);
  EXPECT_EQ(back, host) << "round-trip not byte-exact";
  return out;
}

std::vector<uint8_t> skewedBytes(std::size_t size, uint32_t seed) {
  // Geometric-ish byte distribution: what packed decimal mantissa planes and
  // dictionary codes look like after transforms.
  std::mt19937 gen(seed);
  std::geometric_distribution<int> dist(0.25);
  std::vector<uint8_t> data(size);
  for (auto& byte : data) {
    byte = static_cast<uint8_t>(std::min(dist(gen), 255));
  }
  return data;
}

TEST(UcxCompressionTest, tinyInputSkipped) {
  std::vector<uint8_t> host(1024, 7);
  auto result = roundTrip(host);
  EXPECT_FALSE(result.compressed);
}

TEST(UcxCompressionTest, incompressibleSkipped) {
  std::mt19937 gen(42);
  std::vector<uint8_t> host(8u << 20);
  for (auto& byte : host) {
    byte = static_cast<uint8_t>(gen());
  }
  auto result = roundTrip(host);
  EXPECT_FALSE(result.compressed) << "random bytes must not be compressed";
}

TEST(UcxCompressionTest, skewedSingleSegment) {
  auto result = roundTrip(skewedBytes(8u << 20, 1));
  EXPECT_TRUE(result.compressed);
  EXPECT_GT(result.ratio, 1.5);
}

TEST(UcxCompressionTest, skewedMultiSegmentUnaligned) {
  // Crosses several 32 MiB segments with a ragged tail.
  auto result = roundTrip(skewedBytes((96u << 20) + 12345, 2));
  EXPECT_TRUE(result.compressed);
  EXPECT_GT(result.ratio, 1.5);
  // Local expectation-setting for the B200 runbook: byte-rANS whole-blob
  // should clear a cross-node link on both sides even on this laptop GPU.
  LOG(INFO) << "whole-blob 96MiB: ratio=" << result.ratio
            << " enc=" << result.encodeGbps << " GB/s dec=" << result.decodeGbps
            << " GB/s";
}

TEST(UcxCompressionTest, constantBytes) {
  std::vector<uint8_t> host(40u << 20, 0);
  auto result = roundTrip(host);
  EXPECT_TRUE(result.compressed);
  EXPECT_GT(result.ratio, 50.0);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
