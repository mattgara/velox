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

#include "velox/experimental/ucx-exchange/UcxTransferShaper.h"

#include <future>

#include <gtest/gtest.h>

namespace facebook::velox::ucx_exchange::test {

TEST(UcxTransferShaperTest, serializesAggregateBytes) {
  UcxTransferShaper shaper(1.0, std::chrono::microseconds(100));
  const auto start = UcxTransferShaper::Clock::now();
  const auto first = shaper.reserve(100'000'000);
  const auto second = shaper.reserve(50'000'000);

  EXPECT_GE(first - start, std::chrono::milliseconds(99));
  EXPECT_GE(second - first, std::chrono::milliseconds(49));
  EXPECT_LE(second - first, std::chrono::milliseconds(51));
}

TEST(UcxTransferShaperTest, schedulesWithoutBlockingCaller) {
  UcxTransferShaper shaper(100.0, std::chrono::microseconds(0));
  std::promise<UcxTransferShaper::TimePoint> promise;
  auto future = promise.get_future();
  const auto deadline =
      UcxTransferShaper::Clock::now() + std::chrono::milliseconds(20);

  shaper.scheduleAt(deadline, [&promise] {
    promise.set_value(UcxTransferShaper::Clock::now());
  });

  ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_GE(future.get(), deadline);
}

} // namespace facebook::velox::ucx_exchange::test
