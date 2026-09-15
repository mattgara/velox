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
#include "velox/experimental/ucx-exchange/UcxCompressionCostModel.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace facebook::velox::ucx_exchange::test {

namespace {

void recordSample(
    UcxCompressionCostModel& model,
    std::string_view task,
    std::size_t rawBytes,
    std::size_t encodedBytes,
    double encodeSeconds,
    double transferSeconds,
    double decodeSeconds) {
  model.recordEncode(task, rawBytes, encodedBytes, encodeSeconds);
  model.recordTransfer(task, encodedBytes, transferSeconds);
  model.recordDecode(task, rawBytes, decodeSeconds);
}

} // namespace

TEST(UcxCompressionCostModelTest, SharesSamplesWithinQueryStage) {
  UcxCompressionCostModel model;
  constexpr std::string_view firstTask{"query.4.0.0.0"};
  constexpr std::string_view siblingTask{"query.4.1.0.0"};
  for (int i = 0; i < 4; ++i) {
    recordSample(model, firstTask, 1'000, 400, 0.010, 0.040, 0.010);
  }
  EXPECT_EQ(
      model.decide(siblingTask, 1'000).action,
      UcxCompressionCostModel::Action::kCompress);
  EXPECT_EQ(
      model.decide("query.5.0.0.0", 1'000).action,
      UcxCompressionCostModel::Action::kProbe);
}

TEST(UcxCompressionCostModelTest, FusesWarmupWithNormalCodecSamples) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.4.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // 60% byte savings at 10 kB/s saves 60 ms; the codec costs 20 ms.
    recordSample(model, task, 1'000, 400, 0.010, 0.040, 0.010);
  }
  const auto decision = model.decide(task, 1'000);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kCompress);
  EXPECT_DOUBLE_EQ(decision.encodedRatio, 0.4);
  EXPECT_DOUBLE_EQ(decision.effectiveTransferBytesPerSecond, 10'000.0);
  EXPECT_NEAR(decision.estimatedTransferSavedSeconds, 0.060, 1e-12);
  EXPECT_NEAR(decision.estimatedCodecSeconds, 0.020, 1e-12);
}

TEST(UcxCompressionCostModelTest, SendsRawAndPeriodicallyReprobes) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.3.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // 20 ms transfer savings cannot repay 50 ms of codec work.
    recordSample(model, task, 1'000, 800, 0.030, 0.080, 0.020);
  }
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kRaw);
  }
  EXPECT_EQ(
      model.decide(task, 1'000).action,
      UcxCompressionCostModel::Action::kProbe);
}

TEST(
    UcxCompressionCostModelTest,
    StopsProbingUnsavedBytesWithoutDecodeSamples) {
  UcxCompressionCostModel model(
      /*warmupSamples=*/2,
      /*reprobeInterval=*/3,
      /*codecSafetyMargin=*/1.0,
      /*windowSamples=*/4);
  constexpr std::string_view task{"query.3.0.0"};

  for (int sample = 0; sample < 2; ++sample) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    model.recordEncode(task, 1'000, 1'000, 0.010);
    model.recordTransfer(task, 1'000, 0.040);
  }

  auto decision = model.decide(task, 1'000);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kRaw);
  EXPECT_EQ(decision.decodeSamples, 0);
  EXPECT_DOUBLE_EQ(decision.encodedRatio, 1.0);
  EXPECT_DOUBLE_EQ(decision.estimatedTransferSavedSeconds, 0.0);

  EXPECT_EQ(
      model.decide(task, 1'000).action, UcxCompressionCostModel::Action::kRaw);
  EXPECT_EQ(
      model.decide(task, 1'000).action,
      UcxCompressionCostModel::Action::kProbe);
}

TEST(UcxCompressionCostModelTest, AppliesCodecSafetyMargin) {
  UcxCompressionCostModel model(4, 8, 1.5);
  constexpr std::string_view task{"query.5.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // Saving 60 ms is profitable against 50 ms of codec work without a
    // margin, but not after reserving 50% for opportunity cost.
    recordSample(model, task, 1'000, 400, 0.030, 0.040, 0.020);
  }
  const auto decision = model.decide(task, 1'000);
  EXPECT_NEAR(decision.estimatedTransferSavedSeconds, 0.060, 1e-12);
  EXPECT_NEAR(decision.estimatedCodecSeconds, 0.050, 1e-12);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kRaw);
}

TEST(UcxCompressionCostModelTest, SharesDecodeRateWithinAProcess) {
  UcxCompressionCostModel model;
  constexpr std::string_view incoming{"query.8.0.3.0"};
  constexpr std::string_view outgoing{"query.8.0.0.0"};
  model.recordDecode(incoming, 1'000, 0.010);
  model.recordDecode(incoming, 1'000, 0.010);
  for (int i = 0; i < 4; ++i) {
    model.recordEncode(outgoing, 1'000, 400, 0.010);
    model.recordTransfer(outgoing, 400, 0.040);
  }
  EXPECT_EQ(
      model.decide(outgoing, 1'000).action,
      UcxCompressionCostModel::Action::kCompress);
}

TEST(UcxCompressionCostModelTest, RejectsInvalidSamples) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.9.0.0.0"};
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto infinity = std::numeric_limits<double>::infinity();

  model.recordEncode(task, 1'000, 400, nan);
  model.recordEncode(task, 1'000, 1'001, 0.010);
  model.recordTransfer(task, 400, infinity);
  model.recordDecode(task, 1'000, -0.010);

  const auto decision = model.decide(task, 1'000);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kProbe);
  EXPECT_EQ(decision.encodeSamples, 0);
  EXPECT_EQ(decision.transferSamples, 0);
  EXPECT_EQ(decision.decodeSamples, 0);
}

TEST(UcxCompressionCostModelTest, AccumulatesLargeByteCountsWithoutOverflow) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.10.0.0.0"};
  const auto rawBytes = std::numeric_limits<std::size_t>::max() / 2;
  const auto encodedBytes = rawBytes / 4;

  for (int i = 0; i < 4; ++i) {
    recordSample(model, task, rawBytes, encodedBytes, 0.1, 1.0, 0.1);
  }

  const auto decision = model.decide(task, rawBytes);
  EXPECT_NEAR(decision.encodedRatio, 0.25, 1e-12);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kCompress);
}

TEST(UcxCompressionCostModelTest, BoundsRetainedStageHistory) {
  UcxCompressionCostModel model;
  constexpr std::string_view oldestTask{"oldest.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    recordSample(model, oldestTask, 1'000, 400, 0.010, 0.040, 0.010);
  }
  ASSERT_EQ(
      model.decide(oldestTask, 1'000).action,
      UcxCompressionCostModel::Action::kCompress);

  // Cross the internal 1,024-stage history bound with distinct stage keys.
  constexpr std::size_t kReplacementStageCount = 1024;
  for (std::size_t index = 0; index < kReplacementStageCount; ++index) {
    const auto task = "replacement." + std::to_string(index) + ".0.0";
    model.decide(task, 1'000);
  }

  const auto decision = model.decide(oldestTask, 1'000);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kProbe);
  EXPECT_EQ(decision.encodeSamples, 0);
  EXPECT_EQ(decision.transferSamples, 0);
}
} // namespace facebook::velox::ucx_exchange::test
