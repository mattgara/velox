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

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Result of inspecting a packed-columns metadata buffer. Ordinary cuDF
/// metadata is returned unchanged. Fused-FOR metadata is unwrapped and carries
/// the two scalar values needed to decode its GPU payload.
struct FusedForMetadata {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  bool encoded{false};
  std::size_t segmentCount{0};
  std::size_t logicalDataSize{0};
};

namespace fused_for_wire_detail {

inline constexpr std::array<uint8_t, 8> kMagic{'V', 'L', 'X', 'F',
                                               'F', 'O', 'R', 0};
inline constexpr uint16_t kVersion = 1;
inline constexpr std::size_t kHeaderSize =
    kMagic.size() + sizeof(kVersion) + sizeof(uint16_t) + 3 * sizeof(uint64_t);

template <typename T> void appendScalar(std::vector<uint8_t> &output, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto position = output.size();
  output.resize(position + sizeof(T));
  std::memcpy(output.data() + position, &value, sizeof(T));
}

template <typename T>
T readScalar(const uint8_t *&current, const uint8_t *end) {
  static_assert(std::is_trivially_copyable_v<T>);
  VELOX_CHECK_LE(sizeof(T), static_cast<std::size_t>(end - current),
                 "Truncated fused-FOR metadata envelope");
  T value;
  std::memcpy(&value, current, sizeof(T));
  current += sizeof(T);
  return value;
}

} // namespace fused_for_wire_detail

/// Prepends a small host metadata envelope to ordinary cuDF packed metadata.
/// Tile descriptors remain in the GPU payload and are not serialized here.
inline std::unique_ptr<std::vector<uint8_t>>
wrapFusedForMetadata(std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
                     std::size_t segmentCount, std::size_t logicalDataSize) {
  VELOX_CHECK_NOT_NULL(cudfMetadata);
  using namespace fused_for_wire_detail;
  VELOX_CHECK_LE(cudfMetadata->size(),
                 std::numeric_limits<std::size_t>::max() - kHeaderSize,
                 "Fused-FOR metadata envelope size overflow");

  auto output = std::make_unique<std::vector<uint8_t>>();
  output->reserve(kHeaderSize + cudfMetadata->size());
  output->insert(output->end(), kMagic.begin(), kMagic.end());
  appendScalar(*output, kVersion);
  appendScalar(*output, static_cast<uint16_t>(kHeaderSize));
  appendScalar(*output, static_cast<uint64_t>(segmentCount));
  appendScalar(*output, static_cast<uint64_t>(logicalDataSize));
  appendScalar(*output, static_cast<uint64_t>(cudfMetadata->size()));
  output->insert(output->end(), cudfMetadata->begin(), cudfMetadata->end());
  return output;
}

/// Detects and removes the fused-FOR envelope. A buffer without the magic is
/// ordinary cuDF metadata and is returned without a copy.
inline FusedForMetadata
unwrapFusedForMetadata(std::unique_ptr<std::vector<uint8_t>> metadata) {
  VELOX_CHECK_NOT_NULL(metadata);
  using namespace fused_for_wire_detail;
  if (metadata->size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), metadata->begin())) {
    return FusedForMetadata{std::move(metadata), false, 0, 0};
  }

  VELOX_CHECK_GE(metadata->size(), kHeaderSize,
                 "Truncated fused-FOR metadata envelope");
  const auto *current = metadata->data() + kMagic.size();
  const auto *end = metadata->data() + metadata->size();
  const auto version = readScalar<uint16_t>(current, end);
  const auto headerSize = readScalar<uint16_t>(current, end);
  const auto segmentCount = readScalar<uint64_t>(current, end);
  const auto logicalDataSize = readScalar<uint64_t>(current, end);
  const auto cudfMetadataSize = readScalar<uint64_t>(current, end);

  VELOX_CHECK_EQ(version, kVersion, "Unsupported fused-FOR metadata version");
  VELOX_CHECK_EQ(headerSize, kHeaderSize,
                 "Invalid fused-FOR metadata header size");
  VELOX_CHECK_EQ(cudfMetadataSize, static_cast<uint64_t>(end - current),
                 "Invalid fused-FOR cuDF metadata size");
  VELOX_CHECK_LE(segmentCount, std::numeric_limits<std::size_t>::max(),
                 "Fused-FOR segment count exceeds size_t");
  VELOX_CHECK_LE(logicalDataSize, std::numeric_limits<std::size_t>::max(),
                 "Fused-FOR logical size exceeds size_t");

  auto cudfMetadata = std::make_unique<std::vector<uint8_t>>(
      current, current + static_cast<std::size_t>(cudfMetadataSize));
  return FusedForMetadata{std::move(cudfMetadata), true,
                          static_cast<std::size_t>(segmentCount),
                          static_cast<std::size_t>(logicalDataSize)};
}

} // namespace facebook::velox::ucx_exchange
