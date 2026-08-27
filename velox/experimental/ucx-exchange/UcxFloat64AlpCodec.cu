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

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr uint32_t kNumExponents = 19;
constexpr uint32_t kParameterSlots = kNumExponents * kNumExponents;
constexpr uint32_t kProbeValues = 8u << 10;
constexpr uint32_t kThreads = 256;
constexpr int64_t kExceptionSentinel = LLONG_MIN;
__device__ __constant__ double kDevicePowers[kNumExponents] = {
    1.0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18};
__device__ __constant__ double kDeviceInversePowers[kNumExponents] = {
    1.0,
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
    1e-11,
    1e-12,
    1e-13,
    1e-14,
    1e-15,
    1e-16,
    1e-17,
    1e-18};
constexpr double kEncodingLimit = 9.223372036854774784e18;

inline std::size_t roundUp16(std::size_t value) {
  return (value + 15) & ~static_cast<std::size_t>(15);
}

inline std::size_t packedBytes(uint32_t numValues, uint32_t bitWidth) {
  const std::size_t groups = (static_cast<std::size_t>(numValues) + 31) / 32;
  return groups * bitWidth * sizeof(uint32_t);
}

uint32_t requiredBits(uint64_t range) {
  uint32_t width = 0;
  while (range != 0) {
    ++width;
    range >>= 1;
  }
  return width;
}

#define UCX_ALP_CUDA_CHECK(expr)                      \
  do {                                                \
    cudaError_t error = (expr);                       \
    if (error != cudaSuccess) {                       \
      throw std::runtime_error(                       \
          fmt::format(                                \
              "CUDA error in UCX FP64 ALP codec: {}", \
              cudaGetErrorString(error)));            \
    }                                                 \
  } while (0)

struct ProbeResult {
  int64_t minimum;
  int64_t maximum;
  uint32_t exceptions;
  uint32_t valid;
};

struct FullResult {
  int64_t minimum;
  int64_t maximum;
  uint32_t exceptions;
};

__device__ bool encodeExactly(
    double value,
    uint32_t exponent,
    uint32_t factor,
    int64_t& encoded) {
  if (!isfinite(value)) {
    return false;
  }
  const double scaled =
      value * kDevicePowers[exponent] * kDeviceInversePowers[factor];
  // Match ALP's safe FP64-to-int64 range. LLONG_MIN remains reserved as
  // the exception marker.
  if (!isfinite(scaled) || scaled < -kEncodingLimit ||
      scaled > kEncodingLimit) {
    return false;
  }
  encoded = __double2ll_rn(scaled);
  if (encoded == kExceptionSentinel) {
    return false;
  }
  const double decoded = static_cast<double>(encoded) * kDevicePowers[factor] *
      kDeviceInversePowers[exponent];
  return __double_as_longlong(decoded) == __double_as_longlong(value);
}

__global__ void probeKernel(
    const double* __restrict__ input,
    uint32_t numValues,
    uint32_t sampleCount,
    ProbeResult* __restrict__ results) {
  const uint32_t exponent = blockIdx.x;
  const uint32_t factor = blockIdx.y;
  if (factor > exponent) {
    return;
  }
  int64_t localMinimum = LLONG_MAX;
  int64_t localMaximum = LLONG_MIN;
  uint32_t localExceptions = 0;
  uint32_t localValid = 0;

  for (uint32_t sample = threadIdx.x; sample < sampleCount;
       sample += blockDim.x) {
    const uint32_t index = static_cast<uint32_t>(
        static_cast<uint64_t>(sample) * numValues / sampleCount);
    int64_t encoded = 0;
    if (encodeExactly(input[index], exponent, factor, encoded)) {
      localMinimum = min(localMinimum, encoded);
      localMaximum = max(localMaximum, encoded);
      ++localValid;
    } else {
      ++localExceptions;
    }
  }

  __shared__ int64_t minima[kThreads];
  __shared__ int64_t maxima[kThreads];
  __shared__ uint32_t exceptions[kThreads];
  __shared__ uint32_t valid[kThreads];
  minima[threadIdx.x] = localMinimum;
  maxima[threadIdx.x] = localMaximum;
  exceptions[threadIdx.x] = localExceptions;
  valid[threadIdx.x] = localValid;
  __syncthreads();

  for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      minima[threadIdx.x] =
          min(minima[threadIdx.x], minima[threadIdx.x + width]);
      maxima[threadIdx.x] =
          max(maxima[threadIdx.x], maxima[threadIdx.x + width]);
      exceptions[threadIdx.x] += exceptions[threadIdx.x + width];
      valid[threadIdx.x] += valid[threadIdx.x + width];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    results[exponent * kNumExponents + factor] =
        ProbeResult{minima[0], maxima[0], exceptions[0], valid[0]};
  }
}

__device__ void atomicMinSigned(int64_t* address, int64_t value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (static_cast<int64_t>(old) > value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(value));
    if (old == assumed) {
      break;
    }
  }
}

__device__ void atomicMaxSigned(int64_t* address, int64_t value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (static_cast<int64_t>(old) < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(value));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void transformKernel(
    const double* __restrict__ input,
    int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    uint32_t exponent,
    uint32_t factor,
    FullResult* __restrict__ result) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t encoded = 0;
  const bool inRange = index < numValues;
  const bool valid =
      inRange && encodeExactly(input[index], exponent, factor, encoded);
  if (inRange) {
    encodedValues[index] = valid ? encoded : kExceptionSentinel;
  }

  int64_t localMinimum = valid ? encoded : LLONG_MAX;
  int64_t localMaximum = valid ? encoded : LLONG_MIN;
  uint32_t localExceptions = inRange && !valid ? 1u : 0u;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t mask = __activemask();
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    localMinimum =
        min(localMinimum, __shfl_down_sync(mask, localMinimum, offset));
    localMaximum =
        max(localMaximum, __shfl_down_sync(mask, localMaximum, offset));
    localExceptions += __shfl_down_sync(mask, localExceptions, offset);
  }

  __shared__ int64_t warpMinima[kThreads / 32];
  __shared__ int64_t warpMaxima[kThreads / 32];
  __shared__ uint32_t warpExceptions[kThreads / 32];
  if (lane == 0) {
    warpMinima[warp] = localMinimum;
    warpMaxima[warp] = localMaximum;
    warpExceptions[warp] = localExceptions;
  }
  __syncthreads();

  if (warp == 0) {
    localMinimum = lane < kThreads / 32 ? warpMinima[lane] : LLONG_MAX;
    localMaximum = lane < kThreads / 32 ? warpMaxima[lane] : LLONG_MIN;
    localExceptions = lane < kThreads / 32 ? warpExceptions[lane] : 0;
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      localMinimum =
          min(localMinimum, __shfl_down_sync(mask, localMinimum, offset));
      localMaximum =
          max(localMaximum, __shfl_down_sync(mask, localMaximum, offset));
      localExceptions += __shfl_down_sync(mask, localExceptions, offset);
    }
    if (lane == 0) {
      if (localMinimum != LLONG_MAX) {
        atomicMinSigned(&result->minimum, localMinimum);
        atomicMaxSigned(&result->maximum, localMaximum);
      }
      atomicAdd(&result->exceptions, localExceptions);
    }
  }
}

__global__ void packWarpBitPlanesKernel(
    const int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    int64_t base,
    uint32_t bitWidth,
    uint32_t* __restrict__ packed) {
  const uint32_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t group = globalThread >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t groups = (numValues + 31) / 32;
  if (group >= groups) {
    return;
  }
  const uint32_t index = group * 32 + lane;
  uint64_t adjusted = 0;
  if (index < numValues && encodedValues[index] != kExceptionSentinel) {
    adjusted = static_cast<uint64_t>(encodedValues[index]) -
        static_cast<uint64_t>(base);
  }

  const uint32_t mask = __activemask();
  for (uint32_t bit = 0; bit < bitWidth; ++bit) {
    const uint32_t word = __ballot_sync(mask, (adjusted >> bit) & 1u);
    if (lane == 0) {
      packed[static_cast<std::size_t>(group) * bitWidth + bit] = word;
    }
  }
}

__global__ void compactExceptionsKernel(
    const uint64_t* __restrict__ inputBits,
    const int64_t* __restrict__ encodedValues,
    uint32_t numValues,
    uint32_t* __restrict__ positions,
    uint64_t* __restrict__ values,
    uint32_t* __restrict__ writeCount) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const bool exception =
      index < numValues && encodedValues[index] == kExceptionSentinel;
  const uint32_t mask = __activemask();
  const uint32_t exceptions = __ballot_sync(mask, exception);
  if (exceptions == 0) {
    return;
  }

  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t leader = static_cast<uint32_t>(__ffs(exceptions) - 1);
  uint32_t warpOffset = 0;
  if (lane == leader) {
    warpOffset =
        atomicAdd(writeCount, static_cast<uint32_t>(__popc(exceptions)));
  }
  warpOffset = __shfl_sync(mask, warpOffset, leader);
  if (exception) {
    const uint32_t lowerLanes = lane == 0 ? 0u : ((1u << lane) - 1u);
    const uint32_t slot = warpOffset + __popc(exceptions & lowerLanes);
    positions[slot] = index;
    values[slot] = inputBits[index];
  }
}

__global__ void decodeKernel(
    const uint32_t* __restrict__ packed,
    uint64_t* __restrict__ outputBits,
    uint32_t numValues,
    int64_t base,
    uint32_t bitWidth,
    uint32_t exponent,
    uint32_t factor) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numValues) {
    return;
  }
  const uint32_t group = index >> 5;
  const uint32_t lane = index & 31u;
  uint64_t adjusted = 0;
  for (uint32_t bit = 0; bit < bitWidth; ++bit) {
    const uint32_t word =
        packed[static_cast<std::size_t>(group) * bitWidth + bit];
    adjusted |= static_cast<uint64_t>((word >> lane) & 1u) << bit;
  }
  const int64_t encoded =
      static_cast<int64_t>(static_cast<uint64_t>(base) + adjusted);
  const double decoded = static_cast<double>(encoded) * kDevicePowers[factor] *
      kDeviceInversePowers[exponent];
  outputBits[index] = static_cast<uint64_t>(__double_as_longlong(decoded));
}

__global__ void patchExceptionsKernel(
    const uint32_t* __restrict__ positions,
    const uint64_t* __restrict__ values,
    uint32_t exceptionCount,
    uint64_t* __restrict__ outputBits) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < exceptionCount) {
    outputBits[positions[index]] = values[index];
  }
}

std::pair<uint32_t, uint32_t> selectParameters(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream) {
  const uint32_t sampleCount = std::min(numValues, kProbeValues);
  rmm::device_buffer resultsDevice(
      kParameterSlots * sizeof(ProbeResult), stream);
  const dim3 parameterGrid(kNumExponents, kNumExponents);
  probeKernel<<<parameterGrid, kThreads, 0, stream.value()>>>(
      input,
      numValues,
      sampleCount,
      static_cast<ProbeResult*>(resultsDevice.data()));
  UCX_ALP_CUDA_CHECK(cudaGetLastError());
  std::vector<ProbeResult> results(kParameterSlots);
  UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
      results.data(),
      resultsDevice.data(),
      resultsDevice.size(),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));

  uint32_t selectedExponent = 0;
  uint32_t selectedFactor = 0;
  double bestBytesPerValue = std::numeric_limits<double>::infinity();
  for (uint32_t exponent = 0; exponent < kNumExponents; ++exponent) {
    for (uint32_t factor = 0; factor <= exponent; ++factor) {
      const auto& probe = results[exponent * kNumExponents + factor];
      uint32_t bitWidth = 0;
      if (probe.valid != 0) {
        const uint64_t range = static_cast<uint64_t>(probe.maximum) -
            static_cast<uint64_t>(probe.minimum);
        bitWidth = requiredBits(range);
      }
      const double bytesPerValue = bitWidth / 8.0 +
          static_cast<double>(probe.exceptions) / sampleCount *
              (sizeof(uint32_t) + sizeof(uint64_t));
      if (bytesPerValue < bestBytesPerValue ||
          (bytesPerValue == bestBytesPerValue &&
           (exponent > selectedExponent ||
            (exponent == selectedExponent && factor > selectedFactor)))) {
        bestBytesPerValue = bytesPerValue;
        selectedExponent = exponent;
        selectedFactor = factor;
      }
    }
  }
  return {selectedExponent, selectedFactor};
}

} // namespace

Float64AlpCompressResult compressFloat64Alp(
    const double* input,
    uint32_t numValues,
    rmm::cuda_stream_view stream,
    double minGain) {
  if (minGain < 0.0 || minGain >= 1.0) {
    throw std::invalid_argument("UCX FP64 ALP minGain must be in [0, 1)");
  }

  Float64AlpCompressResult result;
  result.numValues = numValues;
  result.inputBytes = static_cast<std::size_t>(numValues) * sizeof(double);
  if (numValues == 0) {
    result.used = true;
    result.data = rmm::device_buffer(0, stream);
    return result;
  }
  if (input == nullptr) {
    throw std::invalid_argument("UCX FP64 ALP codec received a null input");
  }

  const auto [exponent, factor] = selectParameters(input, numValues, stream);
  result.exponentIndex = exponent;
  result.factorIndex = factor;
  rmm::device_buffer encodedValues(
      static_cast<std::size_t>(numValues) * sizeof(int64_t), stream);
  FullResult initial{LLONG_MAX, LLONG_MIN, 0};
  rmm::device_buffer fullResultDevice(sizeof(FullResult), stream);
  UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
      fullResultDevice.data(),
      &initial,
      sizeof(initial),
      cudaMemcpyHostToDevice,
      stream.value()));
  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  transformKernel<<<blocks, kThreads, 0, stream.value()>>>(
      input,
      static_cast<int64_t*>(encodedValues.data()),
      numValues,
      result.exponentIndex,
      result.factorIndex,
      static_cast<FullResult*>(fullResultDevice.data()));
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  FullResult fullResult;
  UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
      &fullResult,
      fullResultDevice.data(),
      sizeof(fullResult),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.exceptionCount = fullResult.exceptions;
  if (fullResult.exceptions == numValues) {
    result.base = 0;
    result.bitWidth = 0;
  } else {
    result.base = fullResult.minimum;
    const uint64_t range = static_cast<uint64_t>(fullResult.maximum) -
        static_cast<uint64_t>(fullResult.minimum);
    result.bitWidth = requiredBits(range);
  }

  const std::size_t packedSize = packedBytes(numValues, result.bitWidth);
  const std::size_t packedWireSize = roundUp16(packedSize);
  const std::size_t positionsWireSize = roundUp16(
      static_cast<std::size_t>(result.exceptionCount) * sizeof(uint32_t));
  const std::size_t valuesWireSize = roundUp16(
      static_cast<std::size_t>(result.exceptionCount) * sizeof(uint64_t));
  result.candidateBytes = packedWireSize + positionsWireSize + valuesWireSize;
  result.data = rmm::device_buffer(result.candidateBytes, stream);
  if (result.candidateBytes != 0) {
    UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
        result.data.data(), 0, result.data.size(), stream.value()));
  }

  if (result.bitWidth != 0) {
    const uint32_t groups = (numValues + 31) / 32;
    const uint32_t packBlocks = (groups * 32 + kThreads - 1) / kThreads;
    packWarpBitPlanesKernel<<<packBlocks, kThreads, 0, stream.value()>>>(
        static_cast<const int64_t*>(encodedValues.data()),
        numValues,
        result.base,
        result.bitWidth,
        static_cast<uint32_t*>(result.data.data()));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }

  if (result.exceptionCount != 0) {
    auto* positions = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(result.data.data()) + packedWireSize);
    auto* values = reinterpret_cast<uint64_t*>(
        static_cast<uint8_t*>(result.data.data()) + packedWireSize +
        positionsWireSize);
    rmm::device_buffer writeCount(sizeof(uint32_t), stream);
    UCX_ALP_CUDA_CHECK(cudaMemsetAsync(
        writeCount.data(), 0, writeCount.size(), stream.value()));
    compactExceptionsKernel<<<blocks, kThreads, 0, stream.value()>>>(
        reinterpret_cast<const uint64_t*>(input),
        static_cast<const int64_t*>(encodedValues.data()),
        numValues,
        positions,
        values,
        static_cast<uint32_t*>(writeCount.data()));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
    uint32_t written = 0;
    UCX_ALP_CUDA_CHECK(cudaMemcpyAsync(
        &written,
        writeCount.data(),
        sizeof(written),
        cudaMemcpyDeviceToHost,
        stream.value()));
    UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
    if (written != result.exceptionCount) {
      throw std::runtime_error("UCX FP64 ALP exception count mismatch");
    }
  } else {
    UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  }

  result.used = static_cast<double>(result.candidateBytes) <=
      (1.0 - minGain) * result.inputBytes;
  return result;
}

void decompressFloat64AlpPayloadInto(
    const void* data,
    std::size_t candidateBytes,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    uint32_t bitWidth,
    int64_t base,
    double* output,
    rmm::cuda_stream_view stream) {
  if (exponentIndex >= kNumExponents || factorIndex > exponentIndex ||
      bitWidth > 64 || exceptionCount > numValues) {
    throw std::invalid_argument("UCX FP64 ALP metadata is invalid");
  }
  if (numValues == 0) {
    return;
  }
  if ((candidateBytes != 0 && data == nullptr) || output == nullptr) {
    throw std::invalid_argument("UCX FP64 ALP payload or output is null");
  }

  const std::size_t packedSize = packedBytes(numValues, bitWidth);
  const std::size_t packedWireSize = roundUp16(packedSize);
  const std::size_t positionsWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint32_t));
  const std::size_t valuesWireSize =
      roundUp16(static_cast<std::size_t>(exceptionCount) * sizeof(uint64_t));
  if (packedWireSize + positionsWireSize + valuesWireSize != candidateBytes) {
    throw std::invalid_argument("UCX FP64 ALP payload size is invalid");
  }

  const uint32_t blocks = (numValues + kThreads - 1) / kThreads;
  decodeKernel<<<blocks, kThreads, 0, stream.value()>>>(
      static_cast<const uint32_t*>(data),
      reinterpret_cast<uint64_t*>(output),
      numValues,
      base,
      bitWidth,
      exponentIndex,
      factorIndex);
  UCX_ALP_CUDA_CHECK(cudaGetLastError());

  if (exceptionCount != 0) {
    const auto* positions = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(data) + packedWireSize);
    const auto* values = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(data) + packedWireSize + positionsWireSize);
    const uint32_t patchBlocks = (exceptionCount + kThreads - 1) / kThreads;
    patchExceptionsKernel<<<patchBlocks, kThreads, 0, stream.value()>>>(
        positions, values, exceptionCount, reinterpret_cast<uint64_t*>(output));
    UCX_ALP_CUDA_CHECK(cudaGetLastError());
  }
}

rmm::device_buffer decompressFloat64AlpPayload(
    const void* data,
    std::size_t candidateBytes,
    uint32_t numValues,
    uint32_t exceptionCount,
    uint32_t exponentIndex,
    uint32_t factorIndex,
    uint32_t bitWidth,
    int64_t base,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer output(
      static_cast<std::size_t>(numValues) * sizeof(double), stream);
  decompressFloat64AlpPayloadInto(
      data,
      candidateBytes,
      numValues,
      exceptionCount,
      exponentIndex,
      factorIndex,
      bitWidth,
      base,
      static_cast<double*>(output.data()),
      stream);
  UCX_ALP_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return output;
}

rmm::device_buffer decompressFloat64Alp(
    const Float64AlpCompressResult& compressed,
    rmm::cuda_stream_view stream) {
  if (compressed.inputBytes !=
      static_cast<std::size_t>(compressed.numValues) * sizeof(double)) {
    throw std::invalid_argument(
        "UCX FP64 ALP input size does not match its value count");
  }
  return decompressFloat64AlpPayload(
      compressed.data.data(),
      compressed.candidateBytes,
      compressed.numValues,
      compressed.exceptionCount,
      compressed.exponentIndex,
      compressed.factorIndex,
      compressed.bitWidth,
      compressed.base,
      stream);
}

} // namespace facebook::velox::ucx_exchange
