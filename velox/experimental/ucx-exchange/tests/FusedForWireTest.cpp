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

#include "velox/experimental/ucx-exchange/FusedForWire.h"

#include <gtest/gtest.h>

namespace facebook::velox::ucx_exchange {
namespace {

TEST(FusedForWireTest, LeavesOrdinaryMetadataUnchanged) {
  auto metadata = std::make_unique<std::vector<uint8_t>>(
      std::initializer_list<uint8_t>{1, 2, 3, 4, 5});
  const auto *original = metadata.get();

  auto decoded = unwrapFusedForMetadata(std::move(metadata));

  EXPECT_FALSE(decoded.encoded);
  EXPECT_EQ(decoded.cudfMetadata.get(), original);
  EXPECT_EQ(*decoded.cudfMetadata, (std::vector<uint8_t>{1, 2, 3, 4, 5}));
  EXPECT_EQ(decoded.segmentCount, 0);
  EXPECT_EQ(decoded.logicalDataSize, 0);
}

TEST(FusedForWireTest, RoundTripsFusedEnvelope) {
  auto metadata = std::make_unique<std::vector<uint8_t>>(
      std::initializer_list<uint8_t>{9, 7, 5, 3, 1});

  auto wrapped = wrapFusedForMetadata(std::move(metadata), 123, 456'789);
  auto decoded = unwrapFusedForMetadata(std::move(wrapped));

  EXPECT_TRUE(decoded.encoded);
  EXPECT_EQ(decoded.segmentCount, 123);
  EXPECT_EQ(decoded.logicalDataSize, 456'789);
  EXPECT_EQ(*decoded.cudfMetadata, (std::vector<uint8_t>{9, 7, 5, 3, 1}));
}

TEST(FusedForWireTest, RejectsUnsupportedVersion) {
  auto wrapped =
      wrapFusedForMetadata(std::make_unique<std::vector<uint8_t>>(), 1, 32);
  (*wrapped)[fused_for_wire_detail::kMagic.size()] = 2;

  EXPECT_ANY_THROW(unwrapFusedForMetadata(std::move(wrapped)));
}

TEST(FusedForWireTest, RejectsTruncatedEnvelope) {
  auto wrapped =
      wrapFusedForMetadata(std::make_unique<std::vector<uint8_t>>(), 1, 32);
  wrapped->resize(fused_for_wire_detail::kHeaderSize - 1);

  EXPECT_ANY_THROW(unwrapFusedForMetadata(std::move(wrapped)));
}

TEST(FusedForWireTest, RejectsWrongMetadataLength) {
  auto wrapped =
      wrapFusedForMetadata(std::make_unique<std::vector<uint8_t>>(
                               std::initializer_list<uint8_t>{1, 2, 3}),
                           1, 32);
  wrapped->pop_back();

  EXPECT_ANY_THROW(unwrapFusedForMetadata(std::move(wrapped)));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
