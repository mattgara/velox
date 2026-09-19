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

namespace facebook::velox::cudf_velox::compression::detail {

inline constexpr std::size_t kSimpaticoTileRows = 1024;
inline constexpr std::size_t kSimpaticoDecodeGuardBytes = 3 * sizeof(uint32_t);

[[nodiscard]] rmm::device_buffer compressSimpaticoBitpack(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource);

void decompressSimpaticoBitpack(
    cudf::device_span<const uint8_t> input,
    cudf::data_type logicalType,
    cudf::device_span<uint8_t> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource);

[[nodiscard]] std::size_t simpaticoTileCount(
    std::size_t rawSize,
    cudf::data_type logicalType);

[[nodiscard]] std::size_t simpaticoPackedOffset(
    std::size_t rawSize,
    cudf::data_type logicalType);

} // namespace facebook::velox::cudf_velox::compression::detail
