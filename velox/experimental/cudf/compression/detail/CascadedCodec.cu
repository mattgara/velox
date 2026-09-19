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
#include "velox/experimental/cudf/compression/detail/CascadedCodec.h"
#include "velox/experimental/cudf/compression/detail/SizeUtils.h"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

// clang-format off (CudfNoDefaults must follow all cuDF headers)
#include "velox/experimental/cudf/CudfNoDefaults.h"
// clang-format on

#include <nvcomp/cascaded.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::compression::detail {
namespace {

constexpr int kThreadsPerBlock = 256;

[[nodiscard]] nvcompType_t cascadedType(cudf::data_type type) {
  CUDF_EXPECTS(
      cudf::is_fixed_width(type) &&
          (cudf::size_of(type) == 4 || cudf::size_of(type) == 8),
      "nvCOMP Cascaded requires a 4- or 8-byte fixed-width type",
      std::invalid_argument);
  return cudf::size_of(type) == 4 ? NVCOMP_TYPE_INT : NVCOMP_TYPE_LONGLONG;
}

[[nodiscard]] nvcompBatchedCascadedCompressOpts_t compressionOptions(
    cudf::data_type type) {
  auto options = nvcompBatchedCascadedCompressDefaultOpts;
  options.type = cascadedType(type);
  options.num_deltas = 1;
  options.num_RLEs = 0;
  options.use_bp = 1;
  return options;
}

void checkNvcomp(nvcompStatus_t status, const char* operation) {
  CUDF_EXPECTS(
      status == nvcompSuccess,
      std::string{"nvCOMP Cascaded "} + operation + " failed");
}

__global__ void compactChunksKernel(
    const uint8_t* source,
    std::size_t sourceStride,
    uint8_t* destination,
    const std::size_t* destinationOffsets,
    const std::size_t* sizes,
    std::size_t chunkCount) {
  const auto chunk = static_cast<std::size_t>(blockIdx.x);
  if (chunk >= chunkCount) {
    return;
  }
  const auto* input = source + chunk * sourceStride;
  auto* output = destination + destinationOffsets[chunk];
  for (std::size_t offset = threadIdx.x; offset < sizes[chunk];
       offset += blockDim.x) {
    output[offset] = input[offset];
  }
}

struct CompressionMetadata {
  rmm::device_buffer storage;
  const void** inputPointers;
  std::size_t* inputSizes;
  void** outputPointers;
  std::size_t* outputSizes;
  std::size_t* outputOffsets;
  nvcompStatus_t* statuses;
};

[[nodiscard]] CompressionMetadata allocateCompressionMetadata(
    std::size_t chunkCount,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  const auto pointerBytes = checkedMultiplySizes(
      chunkCount, sizeof(void*), "Cascaded pointer metadata size overflow");
  const auto sizeBytes = checkedMultiplySizes(
      chunkCount, sizeof(std::size_t), "Cascaded size metadata overflow");
  const auto statusBytes = checkedMultiplySizes(
      chunkCount, sizeof(nvcompStatus_t), "Cascaded status size overflow");
  const auto totalBytes = checkedAddSizes(
      checkedAddSizes(
          checkedAddSizes(
              checkedAddSizes(
                  pointerBytes, sizeBytes, "Cascaded metadata overflow"),
              pointerBytes,
              "Cascaded metadata overflow"),
          sizeBytes,
          "Cascaded metadata overflow"),
      checkedAddSizes(sizeBytes, statusBytes, "Cascaded metadata overflow"),
      "Cascaded metadata overflow");

  rmm::device_buffer storage{totalBytes, stream, memoryResource};
  auto* cursor = static_cast<uint8_t*>(storage.data());
  auto* inputPointers = reinterpret_cast<const void**>(cursor);
  cursor += pointerBytes;
  auto* inputSizes = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* outputPointers = reinterpret_cast<void**>(cursor);
  cursor += pointerBytes;
  auto* outputSizes = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* outputOffsets = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* statuses = reinterpret_cast<nvcompStatus_t*>(cursor);
  return CompressionMetadata{
      std::move(storage),
      inputPointers,
      inputSizes,
      outputPointers,
      outputSizes,
      outputOffsets,
      statuses};
}

struct DecompressionMetadata {
  rmm::device_buffer storage;
  const void** inputPointers;
  std::size_t* inputSizes;
  std::size_t* outputCapacities;
  std::size_t* actualOutputSizes;
  void** outputPointers;
  nvcompStatus_t* statuses;
};

[[nodiscard]] DecompressionMetadata allocateDecompressionMetadata(
    std::size_t chunkCount,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource) {
  const auto pointerBytes = checkedMultiplySizes(
      chunkCount, sizeof(void*), "Cascaded pointer metadata size overflow");
  const auto sizeBytes = checkedMultiplySizes(
      chunkCount, sizeof(std::size_t), "Cascaded size metadata overflow");
  const auto statusBytes = checkedMultiplySizes(
      chunkCount, sizeof(nvcompStatus_t), "Cascaded status size overflow");
  const auto totalBytes = checkedAddSizes(
      checkedAddSizes(
          checkedAddSizes(
              checkedAddSizes(
                  pointerBytes, sizeBytes, "Cascaded metadata overflow"),
              sizeBytes,
              "Cascaded metadata overflow"),
          sizeBytes,
          "Cascaded metadata overflow"),
      checkedAddSizes(pointerBytes, statusBytes, "Cascaded metadata overflow"),
      "Cascaded metadata overflow");

  rmm::device_buffer storage{totalBytes, stream, memoryResource};
  auto* cursor = static_cast<uint8_t*>(storage.data());
  auto* inputPointers = reinterpret_cast<const void**>(cursor);
  cursor += pointerBytes;
  auto* inputSizes = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* outputCapacities = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* actualOutputSizes = reinterpret_cast<std::size_t*>(cursor);
  cursor += sizeBytes;
  auto* outputPointers = reinterpret_cast<void**>(cursor);
  cursor += pointerBytes;
  auto* statuses = reinterpret_cast<nvcompStatus_t*>(cursor);
  return DecompressionMetadata{
      std::move(storage),
      inputPointers,
      inputSizes,
      outputCapacities,
      actualOutputSizes,
      outputPointers,
      statuses};
}

} // namespace

std::size_t cascadedChunkCount(std::size_t rawSize) noexcept {
  return rawSize / kCascadedChunkBytes + (rawSize % kCascadedChunkBytes != 0);
}

std::size_t cascadedEncodedSize(std::span<const uint32_t> chunkSizes) {
  std::size_t total = 0;
  for (const auto size : chunkSizes) {
    CUDF_EXPECTS(size != 0, "Cascaded chunk size is zero");
    total = checkedAddSizes(
        total, nvcompAlignedSize(size), "Cascaded encoded size overflow");
  }
  return total;
}

CascadedCompressedData compressCascaded(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  CUDF_EXPECTS(!input.empty(), "Cannot compress an empty Cascaded region");
  const auto elementWidth =
      static_cast<std::size_t>(cudf::size_of(logicalType));
  CUDF_EXPECTS(
      input.size() % elementWidth == 0,
      "Cascaded input is not element aligned",
      std::invalid_argument);

  const auto options = compressionOptions(logicalType);
  const auto chunkCount = cascadedChunkCount(input.size());
  CUDF_EXPECTS(
      chunkCount <= std::numeric_limits<unsigned int>::max(),
      "Too many Cascaded chunks",
      std::overflow_error);

  std::vector<const void*> inputPointers(chunkCount);
  std::vector<std::size_t> inputSizes(chunkCount);
  for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
    const auto offset = chunk * kCascadedChunkBytes;
    inputPointers[chunk] = input.data() + offset;
    inputSizes[chunk] = std::min(kCascadedChunkBytes, input.size() - offset);
  }

  std::size_t maximumOutputSize = 0;
  checkNvcomp(
      nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
          kCascadedChunkBytes, options, &maximumOutputSize),
      "maximum-output query");
  const auto outputStride = nvcompAlignedSize(maximumOutputSize);
  rmm::device_buffer scratch{
      checkedMultiplySizes(
          outputStride, chunkCount, "Cascaded scratch size overflow"),
      stream,
      temporaryMemoryResource};
  std::vector<void*> outputPointers(chunkCount);
  for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
    outputPointers[chunk] =
        static_cast<uint8_t*>(scratch.data()) + chunk * outputStride;
  }

  std::size_t temporaryBytes = 0;
  checkNvcomp(
      nvcompBatchedCascadedCompressGetTempSizeAsync(
          chunkCount,
          kCascadedChunkBytes,
          options,
          &temporaryBytes,
          input.size()),
      "temporary-size query");
  rmm::device_buffer temporary{temporaryBytes, stream, temporaryMemoryResource};

  auto metadata =
      allocateCompressionMetadata(chunkCount, stream, temporaryMemoryResource);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.inputPointers,
      inputPointers.data(),
      chunkCount * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.inputSizes,
      inputSizes.data(),
      chunkCount * sizeof(std::size_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.outputPointers,
      outputPointers.data(),
      chunkCount * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream.value()));

  checkNvcomp(
      nvcompBatchedCascadedCompressAsync(
          metadata.inputPointers,
          metadata.inputSizes,
          kCascadedChunkBytes,
          chunkCount,
          temporary.data(),
          temporaryBytes,
          metadata.outputPointers,
          metadata.outputSizes,
          options,
          metadata.statuses,
          stream.value()),
      "compression launch");

  auto stagedSizes =
      cudf::detail::make_pinned_vector_async<std::size_t>(chunkCount, stream);
  auto stagedStatuses = cudf::detail::make_pinned_vector_async<nvcompStatus_t>(
      chunkCount, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedSizes.data(),
      metadata.outputSizes,
      chunkCount * sizeof(std::size_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedStatuses.data(),
      metadata.statuses,
      chunkCount * sizeof(nvcompStatus_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  std::vector<uint32_t> chunkSizes(chunkCount);
  std::vector<std::size_t> outputOffsets(chunkCount);
  std::size_t compressedSize = 0;
  for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
    CUDF_EXPECTS(
        stagedStatuses[chunk] == nvcompSuccess && stagedSizes[chunk] != 0 &&
            stagedSizes[chunk] <= maximumOutputSize &&
            stagedSizes[chunk] <= std::numeric_limits<uint32_t>::max(),
        "nvCOMP Cascaded compression failed");
    chunkSizes[chunk] = static_cast<uint32_t>(stagedSizes[chunk]);
    outputOffsets[chunk] = compressedSize;
    compressedSize = checkedAddSizes(
        compressedSize,
        nvcompAlignedSize(stagedSizes[chunk]),
        "Cascaded output size overflow");
  }

  rmm::device_buffer output{compressedSize, stream, temporaryMemoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(output.data(), 0, output.size(), stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.outputOffsets,
      outputOffsets.data(),
      chunkCount * sizeof(std::size_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  compactChunksKernel<<<
      static_cast<unsigned int>(chunkCount),
      kThreadsPerBlock,
      0,
      stream.value()>>>(
      static_cast<const uint8_t*>(scratch.data()),
      outputStride,
      static_cast<uint8_t*>(output.data()),
      metadata.outputOffsets,
      metadata.outputSizes,
      chunkCount);
  CUDF_CUDA_TRY(cudaGetLastError());
  stream.synchronize();
  return CascadedCompressedData{std::move(output), std::move(chunkSizes)};
}

void decompressCascaded(
    cudf::device_span<const uint8_t> input,
    std::span<const uint32_t> chunkSizes,
    cudf::data_type logicalType,
    cudf::device_span<uint8_t> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  CUDF_EXPECTS(!output.empty(), "Cannot decompress an empty Cascaded region");
  CUDF_EXPECTS(
      input.size() == cascadedEncodedSize(chunkSizes) &&
          chunkSizes.size() == cascadedChunkCount(output.size()),
      "Cascaded descriptor does not match its buffers",
      std::invalid_argument);
  const auto elementWidth =
      static_cast<std::size_t>(cudf::size_of(logicalType));
  CUDF_EXPECTS(
      output.size() % elementWidth == 0,
      "Cascaded output is not element aligned",
      std::invalid_argument);

  const auto chunkCount = chunkSizes.size();
  std::vector<const void*> inputPointers(chunkCount);
  std::vector<std::size_t> inputSizeValues(chunkCount);
  std::vector<std::size_t> outputCapacities(chunkCount);
  std::vector<void*> outputPointers(chunkCount);
  std::size_t inputOffset = 0;
  for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
    inputPointers[chunk] = input.data() + inputOffset;
    inputSizeValues[chunk] = chunkSizes[chunk];
    inputOffset = checkedAddSizes(
        inputOffset,
        nvcompAlignedSize(chunkSizes[chunk]),
        "Cascaded input offset overflow");
    const auto outputOffset = chunk * kCascadedChunkBytes;
    outputCapacities[chunk] =
        std::min(kCascadedChunkBytes, output.size() - outputOffset);
    outputPointers[chunk] = output.data() + outputOffset;
  }

  auto metadata = allocateDecompressionMetadata(
      chunkCount, stream, temporaryMemoryResource);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.inputPointers,
      inputPointers.data(),
      chunkCount * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.inputSizes,
      inputSizeValues.data(),
      chunkCount * sizeof(std::size_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.outputCapacities,
      outputCapacities.data(),
      chunkCount * sizeof(std::size_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      metadata.outputPointers,
      outputPointers.data(),
      chunkCount * sizeof(void*),
      cudaMemcpyHostToDevice,
      stream.value()));

  std::size_t temporaryBytes = 0;
  static_cast<void>(cascadedType(logicalType));
  const auto decompressionOptions = nvcompBatchedCascadedDecompressDefaultOpts;
  checkNvcomp(
      nvcompBatchedCascadedDecompressGetTempSizeAsync(
          chunkCount,
          kCascadedChunkBytes,
          decompressionOptions,
          &temporaryBytes,
          output.size()),
      "decompression temporary-size query");
  rmm::device_buffer temporary{temporaryBytes, stream, temporaryMemoryResource};

  checkNvcomp(
      nvcompBatchedCascadedDecompressAsync(
          metadata.inputPointers,
          metadata.inputSizes,
          metadata.outputCapacities,
          metadata.actualOutputSizes,
          chunkCount,
          temporary.data(),
          temporaryBytes,
          metadata.outputPointers,
          decompressionOptions,
          metadata.statuses,
          stream.value()),
      "decompression launch");

  auto stagedSizes =
      cudf::detail::make_pinned_vector_async<std::size_t>(chunkCount, stream);
  auto stagedStatuses = cudf::detail::make_pinned_vector_async<nvcompStatus_t>(
      chunkCount, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedSizes.data(),
      metadata.actualOutputSizes,
      chunkCount * sizeof(std::size_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      stagedStatuses.data(),
      metadata.statuses,
      chunkCount * sizeof(nvcompStatus_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
    CUDF_EXPECTS(
        stagedStatuses[chunk] == nvcompSuccess &&
            stagedSizes[chunk] == outputCapacities[chunk],
        "nvCOMP Cascaded decompression failed");
  }
}

} // namespace facebook::velox::cudf_velox::compression::detail
