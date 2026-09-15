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

#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

#include "velox/common/base/Exceptions.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace facebook::velox::ucx_exchange {

uint32_t fnv1a_32(std::string_view s) {
  uint32_t hash = 0x811C9DC5u; // FNV offset basis
  for (unsigned char c : s) {
    hash ^= c;
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}

namespace {

std::size_t serializedSize(const MetadataMsg& metadata) {
  std::size_t totalSize = sizeof(kMagicNumber) + sizeof(uint32_t);
  const auto addSize = [&totalSize](std::size_t size) {
    VELOX_CHECK_LE(
        size,
        std::numeric_limits<std::size_t>::max() - totalSize,
        "Metadata serialized size overflow");
    totalSize += size;
  };

  addSize(sizeof(WireLengthType));
  addSize(metadata.cudfMetadata ? metadata.cudfMetadata->size() : 0);
  addSize(sizeof(metadata.dataSizeBytes));
  addSize(sizeof(metadata.numRows));
  addSize(sizeof(WireLengthType));
  VELOX_CHECK_LE(
      metadata.compressionDescriptor.size(),
      std::numeric_limits<std::size_t>::max() /
          sizeof(metadata.compressionDescriptor[0]),
      "Compression descriptor size overflow");
  addSize(
      metadata.compressionDescriptor.size() *
      sizeof(metadata.compressionDescriptor[0]));
  addSize(sizeof(uint8_t));
  return totalSize;
}

} // namespace

std::pair<std::shared_ptr<uint8_t>, std::size_t> MetadataMsg::serialize()
    const {
  const auto requiredSize = serializedSize(*this);
  VELOX_CHECK_GE(dataSizeBytes, 0, "Cannot serialize a negative data size");
  VELOX_CHECK_GE(numRows, 0, "Cannot serialize a negative row count");

  VELOX_CHECK_LE(
      requiredSize,
      kMaxMetaBufSize,
      "Metadata serialized size ({}) exceeds maximum buffer size ({}). "
      "This can happen with extremely wide tables. "
      "Consider reducing table width or increasing kMaxMetaBufSize.",
      requiredSize,
      kMaxMetaBufSize);
  const auto totalSize = static_cast<uint32_t>(requiredSize);

  auto deleter = [](uint8_t* p) { delete[] p; };
  std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);

  uint8_t* ptr = buffer.get();

  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

  const WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
  std::memcpy(ptr, &cudfSize, sizeof(cudfSize));
  ptr += sizeof(cudfSize);

  if (cudfSize > 0) {
    std::memcpy(ptr, cudfMetadata->data(), cudfSize);
    ptr += cudfSize;
  }

  std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
  ptr += sizeof(dataSizeBytes);

  std::memcpy(ptr, &numRows, sizeof(numRows));
  ptr += sizeof(numRows);

  const WireLengthType descriptorWordCount = compressionDescriptor.size();
  std::memcpy(ptr, &descriptorWordCount, sizeof(descriptorWordCount));
  ptr += sizeof(descriptorWordCount);

  if (descriptorWordCount > 0) {
    const auto bytesSize =
        descriptorWordCount * sizeof(compressionDescriptor[0]);
    std::memcpy(ptr, compressionDescriptor.data(), bytesSize);
    ptr += bytesSize;
  }

  const uint8_t atEndByte = atEnd ? 1 : 0;
  *ptr = atEndByte;

  return {std::move(buffer), requiredSize};
}

MetadataMsg MetadataMsg::deserializeMetadataMsg(
    std::span<const uint8_t> buffer) {
  if (buffer.size() < kMetaHeaderSize) {
    throw std::runtime_error("Insufficient data for metadata header");
  }
  const uint8_t* ptr = buffer.data();

  MetadataMsg record;

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
  VELOX_CHECK_EQ(magicNumber, kMagicNumber);
  ptr += sizeof(magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, ptr, sizeof(totalSize));
  ptr += sizeof(totalSize);
  if (totalSize < kMetaHeaderSize || totalSize > buffer.size() ||
      totalSize > kMaxMetaBufSize) {
    throw std::runtime_error("Invalid metadata serialized size");
  }

  const uint8_t* endPtr = buffer.data() + totalSize;
  const auto remaining = [&]() {
    return static_cast<std::size_t>(endPtr - ptr);
  };
  const auto requireBytes = [&](std::size_t size, const char* message) {
    if (size > remaining()) {
      throw std::runtime_error(message);
    }
  };

  WireLengthType metaSize = 0;
  requireBytes(sizeof(metaSize), "Insufficient data for cudfMetadata size");
  std::memcpy(&metaSize, ptr, sizeof(metaSize));
  ptr += sizeof(metaSize);

  if (metaSize > remaining()) {
    throw std::runtime_error("Insufficient data for cudfMetadata bytes");
  }
  const auto metadataSize = static_cast<std::size_t>(metaSize);
  record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metadataSize);
  if (metaSize > 0) {
    std::memcpy(record.cudfMetadata->data(), ptr, metadataSize);
    ptr += metadataSize;
  }

  requireBytes(
      sizeof(record.dataSizeBytes), "Insufficient data for dataSizeBytes");
  std::memcpy(&record.dataSizeBytes, ptr, sizeof(record.dataSizeBytes));
  ptr += sizeof(record.dataSizeBytes);
  if (record.dataSizeBytes < 0) {
    throw std::runtime_error("Negative dataSizeBytes");
  }

  requireBytes(sizeof(record.numRows), "Insufficient data for numRows");
  std::memcpy(&record.numRows, ptr, sizeof(record.numRows));
  ptr += sizeof(record.numRows);
  if (record.numRows < 0) {
    throw std::runtime_error("Negative numRows");
  }

  WireLengthType descriptorWordCount = 0;
  requireBytes(
      sizeof(descriptorWordCount),
      "Insufficient data for compression descriptor count");
  std::memcpy(&descriptorWordCount, ptr, sizeof(descriptorWordCount));
  ptr += sizeof(descriptorWordCount);

  if (descriptorWordCount >
      remaining() / sizeof(record.compressionDescriptor[0])) {
    throw std::runtime_error(
        "Insufficient data for compression descriptor values");
  }
  const auto descriptorSize = static_cast<std::size_t>(descriptorWordCount);
  record.compressionDescriptor.resize(descriptorSize);
  if (descriptorWordCount > 0) {
    const auto bytesSize =
        descriptorSize * sizeof(record.compressionDescriptor[0]);
    std::memcpy(record.compressionDescriptor.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  if (remaining() != sizeof(uint8_t)) {
    throw std::runtime_error("Invalid trailing metadata bytes");
  }
  if (*ptr > 1) {
    throw std::runtime_error("Invalid atEnd flag");
  }
  record.atEnd = (*ptr != 0);

  return record;
}

} // namespace facebook::velox::ucx_exchange
