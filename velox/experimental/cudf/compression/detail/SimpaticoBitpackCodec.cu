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
#include "velox/experimental/cudf/compression/detail/SimpaticoBitpackCodec.h"
#include "velox/experimental/cudf/compression/detail/SizeUtils.h"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

// clang-format off (CudfNoDefaults must follow all cuDF headers)
#include "velox/experimental/cudf/CudfNoDefaults.h"
// clang-format on

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <stdexcept>
#include <type_traits>

namespace facebook::velox::cudf_velox::compression::detail {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kValuesPerThread = kSimpaticoTileRows / kThreadsPerBlock;
static_assert(kValuesPerThread * kThreadsPerBlock == kSimpaticoTileRows);
constexpr uint8_t kDeltaMode = 0x80;
constexpr uint8_t kWidthMask = 0x7f;

template <typename T>
struct Minimum {
  __device__ T operator()(T left, T right) const {
    return left < right ? left : right;
  }
};

template <typename T>
struct Maximum {
  __device__ T operator()(T left, T right) const {
    return left > right ? left : right;
  }
};

__device__ int bitWidth(uint64_t value) {
  return value == 0 ? 0 : 64 - __clzll(value);
}

template <typename T>
__device__ uint64_t adjustedValue(
    const T* input,
    std::size_t tileStart,
    int index,
    bool useDelta,
    T base) {
  using Unsigned = std::make_unsigned_t<T>;
  const auto current = static_cast<Unsigned>(input[tileStart + index]);
  auto transformed = current;
  if (useDelta) {
    const auto previous = tileStart + index == 0
        ? current
        : static_cast<Unsigned>(input[tileStart + index - 1]);
    transformed = current - previous;
  }
  return static_cast<uint64_t>(transformed - static_cast<Unsigned>(base));
}

template <typename T>
__global__ void analyzeKernel(
    const T* input,
    std::size_t elementCount,
    T* tileMinimums,
    T* tileReferences,
    uint8_t* tileBits,
    uint32_t* tileWordCounts,
    SimpaticoTransform requestedTransform) {
  const auto tile = static_cast<std::size_t>(blockIdx.x);
  const auto tileStart = tile * kSimpaticoTileRows;
  if (tileStart >= elementCount) {
    return;
  }
  const auto remainingElements = elementCount - tileStart;
  const auto tileSize = static_cast<int>(
      remainingElements < kSimpaticoTileRows ? remainingElements
                                             : kSimpaticoTileRows);

  using Unsigned = std::make_unsigned_t<T>;
  using SignedDelta = std::make_signed_t<Unsigned>;
  T localMinimum = cuda::std::numeric_limits<T>::max();
  T localMaximum = cuda::std::numeric_limits<T>::lowest();
  SignedDelta localDeltaMinimum = cuda::std::numeric_limits<SignedDelta>::max();
  SignedDelta localDeltaMaximum =
      cuda::std::numeric_limits<SignedDelta>::lowest();
  for (int index = threadIdx.x; index < tileSize; index += blockDim.x) {
    const auto value = input[tileStart + index];
    localMinimum = value < localMinimum ? value : localMinimum;
    localMaximum = value > localMaximum ? value : localMaximum;
    const auto current = static_cast<Unsigned>(value);
    const auto previous = tileStart + index == 0
        ? current
        : static_cast<Unsigned>(input[tileStart + index - 1]);
    const auto delta = cuda::std::bit_cast<SignedDelta>(current - previous);
    localDeltaMinimum = delta < localDeltaMinimum ? delta : localDeltaMinimum;
    localDeltaMaximum = delta > localDeltaMaximum ? delta : localDeltaMaximum;
  }

  using ValueReduction = cub::BlockReduce<T, kThreadsPerBlock>;
  using DeltaReduction = cub::BlockReduce<SignedDelta, kThreadsPerBlock>;
  __shared__ typename ValueReduction::TempStorage valueReductionStorage;
  __shared__ typename DeltaReduction::TempStorage deltaReductionStorage;
  __shared__ T sharedMinimum;
  __shared__ T sharedMaximum;
  __shared__ SignedDelta sharedDeltaMinimum;
  __shared__ SignedDelta sharedDeltaMaximum;
  const auto minimum =
      ValueReduction(valueReductionStorage).Reduce(localMinimum, Minimum<T>{});
  if (threadIdx.x == 0) {
    sharedMinimum = minimum;
  }
  __syncthreads();
  const auto maximum =
      ValueReduction(valueReductionStorage).Reduce(localMaximum, Maximum<T>{});
  if (threadIdx.x == 0) {
    sharedMaximum = maximum;
  }
  __syncthreads();
  const auto deltaMinimum =
      DeltaReduction(deltaReductionStorage)
          .Reduce(localDeltaMinimum, Minimum<SignedDelta>{});
  if (threadIdx.x == 0) {
    sharedDeltaMinimum = deltaMinimum;
  }
  __syncthreads();
  const auto deltaMaximum =
      DeltaReduction(deltaReductionStorage)
          .Reduce(localDeltaMaximum, Maximum<SignedDelta>{});
  if (threadIdx.x == 0) {
    sharedDeltaMaximum = deltaMaximum;
  }
  __syncthreads();

  const auto frameRange = static_cast<uint64_t>(
      static_cast<Unsigned>(sharedMaximum) -
      static_cast<Unsigned>(sharedMinimum));
  const auto deltaRange = static_cast<uint64_t>(
      static_cast<Unsigned>(sharedDeltaMaximum) -
      static_cast<Unsigned>(sharedDeltaMinimum));
  const auto frameBits = bitWidth(frameRange);
  const auto deltaBits = bitWidth(deltaRange);
  const auto useDelta =
      requestedTransform == SimpaticoTransform::kDeltaFrameOfReference ||
      (requestedTransform == SimpaticoTransform::kAutomatic &&
       deltaBits < frameBits);
  const auto base =
      useDelta ? cuda::std::bit_cast<T>(sharedDeltaMinimum) : sharedMinimum;
  const auto bits = useDelta ? deltaBits : frameBits;
  const auto liveWords =
      static_cast<uint32_t>((static_cast<uint64_t>(tileSize) * bits + 31) / 32);
  if (threadIdx.x == 0) {
    tileMinimums[tile] = base;
    tileReferences[tile] = tileStart == 0 ? input[0] : input[tileStart - 1];
    tileBits[tile] =
        static_cast<uint8_t>(bits) | (useDelta ? kDeltaMode : uint8_t{0});
    tileWordCounts[tile] = liveWords;
  }
}

template <typename T>
__global__ void packKernel(
    const T* input,
    std::size_t elementCount,
    const T* tileMinimums,
    const uint8_t* tileBits,
    const uint32_t* offsets,
    uint32_t* packed) {
  const auto tile = static_cast<std::size_t>(blockIdx.x);
  const auto tileStart = tile * kSimpaticoTileRows;
  if (tileStart >= elementCount) {
    return;
  }
  const auto remainingElements = elementCount - tileStart;
  const auto tileSize = static_cast<int>(
      remainingElements < kSimpaticoTileRows ? remainingElements
                                             : kSimpaticoTileRows);
  const auto encodedBits = tileBits[tile];
  const auto bits = static_cast<int>(encodedBits & kWidthMask);
  if (bits == 0) {
    return;
  }

  const auto useDelta = (encodedBits & kDeltaMode) != 0;
  const auto base = tileMinimums[tile];
  auto* destination = packed + offsets[tile];
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    if (bits == 32) {
      for (int index = threadIdx.x; index < tileSize; index += blockDim.x) {
        destination[index] = static_cast<uint32_t>(
            adjustedValue(input, tileStart, index, useDelta, base));
      }
      return;
    }
  }
  if constexpr (sizeof(T) == sizeof(uint64_t)) {
    if (bits == 64) {
      for (int index = threadIdx.x; index < tileSize; index += blockDim.x) {
        const auto adjusted =
            adjustedValue(input, tileStart, index, useDelta, base);
        destination[2 * index] = static_cast<uint32_t>(adjusted);
        destination[2 * index + 1] = static_cast<uint32_t>(adjusted >> 32);
      }
      return;
    }
  }

  for (int index = threadIdx.x; index < tileSize; index += blockDim.x) {
    const auto adjusted =
        adjustedValue(input, tileStart, index, useDelta, base);
    auto remaining = bits;
    auto sourceBit = 0;
    auto targetWord = (index * bits) >> 5;
    auto targetBit = (index * bits) & 31;
    while (remaining > 0) {
      const auto availableBits = 32 - targetBit;
      const auto count = availableBits < remaining ? availableBits : remaining;
      const auto mask =
          count == 32 ? uint32_t{0xffffffff} : (uint32_t{1} << count) - 1;
      atomicOr(
          destination + targetWord,
          static_cast<uint32_t>((adjusted >> sourceBit) & mask) << targetBit);
      sourceBit += count;
      remaining -= count;
      ++targetWord;
      targetBit = 0;
    }
  }
}

__global__ void finishOffsetsKernel(
    const uint32_t* wordCounts,
    uint32_t* offsets,
    std::size_t tileCount) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    offsets[tileCount] =
        tileCount == 0 ? 0 : offsets[tileCount - 1] + wordCounts[tileCount - 1];
  }
}

__global__ void deriveWordCountsKernel(
    const uint8_t* tileBits,
    std::size_t elementCount,
    uint32_t* wordCounts,
    uint32_t* invalid,
    std::size_t tileCount,
    int maximumBits) {
  const auto tile =
      static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tile >= tileCount) {
    return;
  }
  const auto bits = static_cast<int>(tileBits[tile] & kWidthMask);
  if (bits > maximumBits) {
    atomicExch(invalid, 1);
    wordCounts[tile] = 0;
    return;
  }
  const auto tileStart = tile * kSimpaticoTileRows;
  const auto remainingElements = elementCount - tileStart;
  const auto count = remainingElements < kSimpaticoTileRows
      ? remainingElements
      : kSimpaticoTileRows;
  wordCounts[tile] =
      static_cast<uint32_t>((static_cast<uint64_t>(count) * bits + 31) / 32);
}

__device__ uint64_t
unpackOne(const uint32_t* packed, int bits, uint32_t index) {
  const auto bitPosition = index * static_cast<uint32_t>(bits);
  const auto word = bitPosition >> 5;
  const auto bit = static_cast<int>(bitPosition & 31);
  const auto shift = 32 - bit;
  const uint64_t word0 = packed[word];
  const uint64_t word1 = packed[word + 1];
  const uint64_t word2 = packed[word + 2];
  const auto low = ((word0 >> bit) | (word1 << shift)) & 0xffffffffULL;
  const auto high = ((word1 >> bit) | (word2 << shift)) & 0xffffffffULL;
  const auto value = (high << 32) | low;
  return bits == 64 ? value : value & ((uint64_t{1} << bits) - 1);
}

template <typename T>
__global__ void unpackKernel(
    const T* tileMinimums,
    const T* tileReferences,
    const uint8_t* tileBits,
    const uint32_t* offsets,
    const uint32_t* packed,
    std::size_t elementCount,
    T* output) {
  using Unsigned = std::make_unsigned_t<T>;
  const auto tile = static_cast<std::size_t>(blockIdx.x);
  const auto tileStart = tile * kSimpaticoTileRows;
  if (tileStart >= elementCount) {
    return;
  }
  const auto remainingElements = elementCount - tileStart;
  const auto tileSize = remainingElements < kSimpaticoTileRows
      ? remainingElements
      : kSimpaticoTileRows;
  const auto encodedBits = tileBits[tile];
  const auto bits = static_cast<int>(encodedBits & kWidthMask);
  const auto useDelta = (encodedBits & kDeltaMode) != 0;
  const auto base = static_cast<Unsigned>(tileMinimums[tile]);
  const auto* input = packed + offsets[tile];
  auto* unsignedOutput = reinterpret_cast<Unsigned*>(output);

  if (!useDelta) {
    for (std::size_t index = threadIdx.x; index < tileSize;
         index += blockDim.x) {
      const auto adjusted =
          bits == 0 ? uint64_t{0} : unpackOne(input, bits, index);
      unsignedOutput[tileStart + index] =
          base + static_cast<Unsigned>(adjusted);
    }
    return;
  }

  Unsigned values[kValuesPerThread];
#pragma unroll
  for (int value = 0; value < kValuesPerThread; ++value) {
    const auto index = threadIdx.x * kValuesPerThread + value;
    const auto adjusted = index < tileSize && bits != 0
        ? unpackOne(input, bits, index)
        : uint64_t{0};
    values[value] =
        index < tileSize ? base + static_cast<Unsigned>(adjusted) : Unsigned{0};
  }

  using Scan = cub::BlockScan<Unsigned, kThreadsPerBlock>;
  __shared__ typename Scan::TempStorage scanStorage;
  __shared__ Unsigned transposed[kSimpaticoTileRows];
  Scan(scanStorage).InclusiveSum(values, values);
  const auto reference = static_cast<Unsigned>(tileReferences[tile]);
#pragma unroll
  for (int value = 0; value < kValuesPerThread; ++value) {
    const auto index = threadIdx.x * kValuesPerThread + value;
    if (index < tileSize) {
      transposed[index] = reference + values[value];
    }
  }
  __syncthreads();
#pragma unroll
  for (int value = 0; value < kValuesPerThread; ++value) {
    const auto index = value * blockDim.x + threadIdx.x;
    if (index < tileSize) {
      unsignedOutput[tileStart + index] = transposed[index];
    }
  }
}

[[nodiscard]] bool usesUnsignedStorage(cudf::data_type type) noexcept {
  return type.id() == cudf::type_id::UINT32 ||
      type.id() == cudf::type_id::UINT64;
}

[[nodiscard]] std::size_t referenceOffset(
    std::size_t rawSize,
    cudf::data_type type) {
  const auto tiles = simpaticoTileCount(rawSize, type);
  return nvcompAlignedSize(checkedMultiplySizes(
      tiles,
      static_cast<std::size_t>(cudf::size_of(type)),
      "Simpatico minimum array size overflow"));
}

[[nodiscard]] std::size_t bitsOffset(
    std::size_t rawSize,
    cudf::data_type type) {
  const auto tiles = simpaticoTileCount(rawSize, type);
  const auto referencesEnd = checkedAddSizes(
      referenceOffset(rawSize, type),
      checkedMultiplySizes(
          tiles,
          static_cast<std::size_t>(cudf::size_of(type)),
          "Simpatico reference array size overflow"),
      "Simpatico metadata size overflow");
  return nvcompAlignedSize(referencesEnd);
}

template <typename T>
[[nodiscard]] rmm::device_buffer compressTyped(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    SimpaticoTransform transform,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  const auto elementCount = input.size() / sizeof(T);
  const auto tileCount = simpaticoTileCount(input.size(), logicalType);

  rmm::device_buffer minimums{
      checkedMultiplySizes(
          tileCount, sizeof(T), "Simpatico minimum array size overflow"),
      stream,
      memoryResource};
  rmm::device_buffer references{
      checkedMultiplySizes(
          tileCount, sizeof(T), "Simpatico reference array size overflow"),
      stream,
      memoryResource};
  rmm::device_buffer bits{tileCount, stream, memoryResource};
  rmm::device_buffer wordCounts{
      checkedMultiplySizes(
          tileCount, sizeof(uint32_t), "Simpatico word-count size overflow"),
      stream,
      memoryResource};
  rmm::device_buffer offsets{
      checkedMultiplySizes(
          tileCount + 1, sizeof(uint32_t), "Simpatico offset size overflow"),
      stream,
      memoryResource};

  analyzeKernel<T>
      <<<static_cast<unsigned int>(tileCount),
         kThreadsPerBlock,
         0,
         stream.value()>>>(
          reinterpret_cast<const T*>(input.data()),
          elementCount,
          static_cast<T*>(minimums.data()),
          static_cast<T*>(references.data()),
          static_cast<uint8_t*>(bits.data()),
          static_cast<uint32_t*>(wordCounts.data()),
          transform);
  CUDF_CUDA_TRY(cudaGetLastError());

  std::size_t scanBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(
          nullptr,
          scanBytes,
          static_cast<const uint32_t*>(wordCounts.data()),
          static_cast<uint32_t*>(offsets.data()),
          tileCount,
          stream.value()));
  rmm::device_buffer scanTemporary{scanBytes, stream, memoryResource};
  CUDF_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(
          scanTemporary.data(),
          scanBytes,
          static_cast<const uint32_t*>(wordCounts.data()),
          static_cast<uint32_t*>(offsets.data()),
          tileCount,
          stream.value()));
  finishOffsetsKernel<<<1, 1, 0, stream.value()>>>(
      static_cast<const uint32_t*>(wordCounts.data()),
      static_cast<uint32_t*>(offsets.data()),
      tileCount);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto stagedLiveWords =
      cudf::detail::make_pinned_vector_async<uint32_t>(1, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedLiveWords.data(),
      static_cast<const uint32_t*>(offsets.data()) + tileCount,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  const auto liveBytes = checkedMultiplySizes(
      stagedLiveWords[0],
      sizeof(uint32_t),
      "Simpatico compact payload size overflow");
  const auto packedOffset = simpaticoPackedOffset(input.size(), logicalType);
  const auto outputSize = checkedAddSizes(
      checkedAddSizes(
          packedOffset, liveBytes, "Simpatico payload size overflow"),
      kSimpaticoDecodeGuardBytes,
      "Simpatico payload size overflow");
  rmm::device_buffer output{outputSize, stream, memoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(output.data(), 0, output.size(), stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      output.data(),
      minimums.data(),
      minimums.size(),
      cudaMemcpyDeviceToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      static_cast<uint8_t*>(output.data()) +
          referenceOffset(input.size(), logicalType),
      references.data(),
      references.size(),
      cudaMemcpyDeviceToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      static_cast<uint8_t*>(output.data()) +
          bitsOffset(input.size(), logicalType),
      bits.data(),
      bits.size(),
      cudaMemcpyDeviceToDevice,
      stream.value()));
  packKernel<T>
      <<<static_cast<unsigned int>(tileCount),
         kThreadsPerBlock,
         0,
         stream.value()>>>(
          reinterpret_cast<const T*>(input.data()),
          elementCount,
          static_cast<const T*>(minimums.data()),
          static_cast<const uint8_t*>(bits.data()),
          static_cast<const uint32_t*>(offsets.data()),
          reinterpret_cast<uint32_t*>(
              static_cast<uint8_t*>(output.data()) + packedOffset));
  CUDF_CUDA_TRY(cudaGetLastError());
  stream.synchronize();
  return output;
}

template <typename T>
void decompressTyped(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    cudf::device_span<uint8_t> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  const auto elementCount = output.size() / sizeof(T);
  const auto tileCount = simpaticoTileCount(output.size(), logicalType);
  const auto packedOffset = simpaticoPackedOffset(output.size(), logicalType);
  CUDF_EXPECTS(
      input.size() >= packedOffset + kSimpaticoDecodeGuardBytes &&
          (input.size() - packedOffset - kSimpaticoDecodeGuardBytes) %
                  sizeof(uint32_t) ==
              0,
      "Invalid Simpatico bitpack payload size",
      std::invalid_argument);
  const auto liveWords =
      (input.size() - packedOffset - kSimpaticoDecodeGuardBytes) /
      sizeof(uint32_t);

  rmm::device_buffer wordCounts{
      checkedMultiplySizes(
          tileCount, sizeof(uint32_t), "Simpatico word-count size overflow"),
      stream,
      memoryResource};
  rmm::device_buffer offsets{
      checkedMultiplySizes(
          tileCount + 1, sizeof(uint32_t), "Simpatico offset size overflow"),
      stream,
      memoryResource};
  rmm::device_buffer invalid{sizeof(uint32_t), stream, memoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(invalid.data(), 0, invalid.size(), stream.value()));

  constexpr int kValidationThreads = 256;
  const auto validationBlocks = static_cast<unsigned int>(
      (tileCount + kValidationThreads - 1) / kValidationThreads);
  deriveWordCountsKernel<<<
      validationBlocks,
      kValidationThreads,
      0,
      stream.value()>>>(
      input.data() + bitsOffset(output.size(), logicalType),
      elementCount,
      static_cast<uint32_t*>(wordCounts.data()),
      static_cast<uint32_t*>(invalid.data()),
      tileCount,
      sizeof(T) * 8);
  CUDF_CUDA_TRY(cudaGetLastError());

  std::size_t scanBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(
          nullptr,
          scanBytes,
          static_cast<const uint32_t*>(wordCounts.data()),
          static_cast<uint32_t*>(offsets.data()),
          tileCount,
          stream.value()));
  rmm::device_buffer scanTemporary{scanBytes, stream, memoryResource};
  CUDF_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(
          scanTemporary.data(),
          scanBytes,
          static_cast<const uint32_t*>(wordCounts.data()),
          static_cast<uint32_t*>(offsets.data()),
          tileCount,
          stream.value()));
  finishOffsetsKernel<<<1, 1, 0, stream.value()>>>(
      static_cast<const uint32_t*>(wordCounts.data()),
      static_cast<uint32_t*>(offsets.data()),
      tileCount);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto stagedValidation =
      cudf::detail::make_pinned_vector_async<uint32_t>(2, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedValidation.data(),
      invalid.data(),
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedValidation.data() + 1,
      static_cast<const uint32_t*>(offsets.data()) + tileCount,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  CUDF_EXPECTS(
      stagedValidation[0] == 0 && stagedValidation[1] == liveWords,
      "Simpatico bitpack metadata does not match its payload",
      std::invalid_argument);

  unpackKernel<T>
      <<<static_cast<unsigned int>(tileCount),
         kThreadsPerBlock,
         0,
         stream.value()>>>(
          reinterpret_cast<const T*>(input.data()),
          reinterpret_cast<const T*>(
              input.data() + referenceOffset(output.size(), logicalType)),
          input.data() + bitsOffset(output.size(), logicalType),
          static_cast<const uint32_t*>(offsets.data()),
          reinterpret_cast<const uint32_t*>(input.data() + packedOffset),
          elementCount,
          reinterpret_cast<T*>(output.data()));
  CUDF_CUDA_TRY(cudaGetLastError());
  stream.synchronize();
}

} // namespace

std::size_t simpaticoTileCount(
    std::size_t rawSize,
    cudf::data_type logicalType) {
  CUDF_EXPECTS(
      cudf::is_fixed_width(logicalType) &&
          (cudf::size_of(logicalType) == 4 || cudf::size_of(logicalType) == 8),
      "Simpatico bitpack requires a 4- or 8-byte fixed-width type",
      std::invalid_argument);
  const auto width = static_cast<std::size_t>(cudf::size_of(logicalType));
  CUDF_EXPECTS(
      rawSize != 0 && rawSize % width == 0,
      "Simpatico region is not element aligned",
      std::invalid_argument);
  const auto elements = rawSize / width;
  return elements / kSimpaticoTileRows + (elements % kSimpaticoTileRows != 0);
}

std::size_t simpaticoPackedOffset(
    std::size_t rawSize,
    cudf::data_type logicalType) {
  const auto tiles = simpaticoTileCount(rawSize, logicalType);
  return nvcompAlignedSize(checkedAddSizes(
      bitsOffset(rawSize, logicalType),
      tiles,
      "Simpatico metadata size overflow"));
}

rmm::device_buffer compressSimpaticoBitpack(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    SimpaticoTransform transform,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  if (cudf::size_of(logicalType) == 8) {
    if (usesUnsignedStorage(logicalType)) {
      return compressTyped<uint64_t>(
          input, logicalType, transform, stream, temporaryMemoryResource);
    }
    return compressTyped<int64_t>(
        input, logicalType, transform, stream, temporaryMemoryResource);
  }
  if (usesUnsignedStorage(logicalType)) {
    return compressTyped<uint32_t>(
        input, logicalType, transform, stream, temporaryMemoryResource);
  }
  return compressTyped<int32_t>(
      input, logicalType, transform, stream, temporaryMemoryResource);
}

void decompressSimpaticoBitpack(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    cudf::device_span<uint8_t> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  if (cudf::size_of(logicalType) == 8) {
    if (usesUnsignedStorage(logicalType)) {
      decompressTyped<uint64_t>(
          input, logicalType, output, stream, temporaryMemoryResource);
    } else {
      decompressTyped<int64_t>(
          input, logicalType, output, stream, temporaryMemoryResource);
    }
    return;
  }
  if (usesUnsignedStorage(logicalType)) {
    decompressTyped<uint32_t>(
        input, logicalType, output, stream, temporaryMemoryResource);
  } else {
    decompressTyped<int32_t>(
        input, logicalType, output, stream, temporaryMemoryResource);
  }
}

} // namespace facebook::velox::cudf_velox::compression::detail
