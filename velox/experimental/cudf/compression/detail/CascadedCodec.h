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
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::compression::detail {

// This matches the winning standalone nvCOMP Cascaded experiment. Each
// independent frame sees at most 64 KiB of one typed region.
inline constexpr std::size_t kCascadedChunkBytes = 64u << 10;

struct CascadedCompressedData {
  rmm::device_buffer data;
  std::vector<uint32_t> chunkSizes;
};

/**
 * Compresses one 4- or 8-byte fixed-width region with one delta, no RLE, and
 * bitpacking. Chunks are stored consecutively at nvCOMP-aligned offsets.
 */
[[nodiscard]] CascadedCompressedData compressCascaded(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource);

/** Reconstructs a region produced by compressCascaded. */
void decompressCascaded(
    cudf::device_span<const uint8_t> input,
    std::span<const uint32_t> chunkSizes,
    cudf::data_type logicalType,
    cudf::device_span<uint8_t> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource);

[[nodiscard]] std::size_t cascadedEncodedSize(
    std::span<const uint32_t> chunkSizes);

[[nodiscard]] std::size_t cascadedChunkCount(std::size_t rawSize) noexcept;

} // namespace facebook::velox::cudf_velox::compression::detail
