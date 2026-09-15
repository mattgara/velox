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

#include <algorithm>
#include <cmath>
#include <numeric>

namespace facebook::velox::ucx_exchange {

namespace {
// Query identifiers are process-lifetime values. Bound retained stage history
// so a long-lived worker does not accumulate one entry per completed query.
constexpr std::size_t kMaximumTrackedStages = 1024;

constexpr std::size_t kMinimumDecodeSamples = 2;

template <typename Samples>
long double sumBytes(const Samples& samples) {
  return std::accumulate(
      samples.begin(),
      samples.end(),
      0.0L,
      [](long double total, const auto& sample) {
        return total + static_cast<long double>(sample.bytes);
      });
}

template <typename Samples>
long double sumSeconds(const Samples& samples) {
  return std::accumulate(
      samples.begin(),
      samples.end(),
      0.0L,
      [](long double total, const auto& sample) {
        return total + static_cast<long double>(sample.seconds);
      });
}

} // namespace

UcxCompressionCostModel::UcxCompressionCostModel(
    std::size_t warmupSamples,
    std::size_t reprobeInterval,
    double codecSafetyMargin,
    std::size_t windowSamples)
    : warmupSamples_(std::max<std::size_t>(1, warmupSamples)),
      reprobeInterval_(std::max<std::size_t>(1, reprobeInterval)),
      codecSafetyMargin_(std::max(1.0, codecSafetyMargin)),
      windowSamples_(
          std::max({std::size_t{1}, windowSamples, warmupSamples_})) {}

UcxCompressionCostModel& UcxCompressionCostModel::instance(
    double codecSafetyMargin) {
  static UcxCompressionCostModel model(4, 8, codecSafetyMargin);
  return model;
}

UcxCompressionCostModel::StageSamples&
UcxCompressionCostModel::getOrCreateStage(std::string_view taskId) {
  auto key = stageKey(taskId);
  if (auto existing = stages_.find(key); existing != stages_.end()) {
    return existing->second;
  }

  if (stages_.size() >= kMaximumTrackedStages) {
    stages_.erase(stageOrder_.front());
    stageOrder_.pop_front();
  }

  stageOrder_.push_back(key);
  return stages_.try_emplace(std::move(key)).first->second;
}

UcxCompressionCostModel::Decision UcxCompressionCostModel::decide(
    std::string_view taskId,
    std::size_t rawBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& stage = getOrCreateStage(taskId);

  Decision decision;
  decision.encodeSamples = stage.encodes.size();
  decision.transferSamples = stage.transfers.size();

  const auto& decodes = stage.decodes.size() >= kMinimumDecodeSamples
      ? stage.decodes
      : globalDecodes_;
  decision.decodeSamples = decodes.size();

  if (stage.encodes.size() < warmupSamples_ ||
      stage.transfers.size() < warmupSamples_ || rawBytes == 0) {
    stage.rawDecisionsSinceProbe = 0;
    return decision;
  }

  long double encodedRawBytes = 0.0L;
  long double encodedBytes = 0.0L;
  long double encodeSeconds = 0.0L;
  for (const auto& sample : stage.encodes) {
    encodedRawBytes += static_cast<long double>(sample.rawBytes);
    encodedBytes += static_cast<long double>(sample.encodedBytes);
    encodeSeconds += static_cast<long double>(sample.seconds);
  }
  const auto transferredBytes = sumBytes(stage.transfers);
  const auto transferSeconds = sumSeconds(stage.transfers);

  if (encodedRawBytes == 0 || transferredBytes == 0 || transferSeconds <= 0.0 ||
      encodeSeconds <= 0.0) {
    stage.rawDecisionsSinceProbe = 0;
    return decision;
  }

  const auto encodedRatio = encodedBytes / encodedRawBytes;
  const auto transferBytesPerSecond = transferredBytes / transferSeconds;
  const auto savedBytes = std::max(
      0.0L, static_cast<long double>(rawBytes) * (1.0L - encodedRatio));
  auto codecSeconds =
      static_cast<long double>(rawBytes) * encodeSeconds / encodedRawBytes;

  // A probe that produced no byte saving cannot benefit from a decode, and no
  // receiver decode sample exists because that candidate was sent raw.
  if (savedBytes > 0.0L) {
    const auto decodedBytes = sumBytes(decodes);
    const auto decodeSeconds = sumSeconds(decodes);
    if (decodes.size() < kMinimumDecodeSamples || decodedBytes == 0 ||
        decodeSeconds <= 0.0) {
      stage.rawDecisionsSinceProbe = 0;
      return decision;
    }
    codecSeconds +=
        static_cast<long double>(rawBytes) * decodeSeconds / decodedBytes;
  }

  decision.encodedRatio = static_cast<double>(encodedRatio);
  decision.effectiveTransferBytesPerSecond =
      static_cast<double>(transferBytesPerSecond);
  decision.estimatedTransferSavedSeconds =
      static_cast<double>(savedBytes / transferBytesPerSecond);
  decision.estimatedCodecSeconds = static_cast<double>(codecSeconds);

  if (decision.estimatedTransferSavedSeconds >
      codecSafetyMargin_ * decision.estimatedCodecSeconds) {
    stage.rawDecisionsSinceProbe = 0;
    decision.action = Action::kCompress;
    return decision;
  }

  ++stage.rawDecisionsSinceProbe;
  if (stage.rawDecisionsSinceProbe >= reprobeInterval_) {
    stage.rawDecisionsSinceProbe = 0;
    decision.action = Action::kProbe;
    return decision;
  }
  decision.action = Action::kRaw;
  return decision;
}

void UcxCompressionCostModel::recordEncode(
    std::string_view taskId,
    std::size_t rawBytes,
    std::size_t encodedBytes,
    double seconds) {
  if (rawBytes == 0 || encodedBytes == 0 || encodedBytes > rawBytes ||
      !std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  append(
      getOrCreateStage(taskId).encodes,
      EncodeSample{rawBytes, encodedBytes, seconds});
}

void UcxCompressionCostModel::recordTransfer(
    std::string_view taskId,
    std::size_t wireBytes,
    double seconds) {
  if (wireBytes == 0 || !std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  append(
      getOrCreateStage(taskId).transfers, ByteTimeSample{wireBytes, seconds});
}

void UcxCompressionCostModel::recordDecode(
    std::string_view remoteTaskId,
    std::size_t rawBytes,
    double seconds) {
  if (rawBytes == 0 || !std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const ByteTimeSample sample{rawBytes, seconds};
  append(getOrCreateStage(remoteTaskId).decodes, sample);
  append(globalDecodes_, sample);
}

std::string UcxCompressionCostModel::stageKey(std::string_view taskId) {
  const auto first = taskId.find('.');
  if (first == std::string_view::npos) {
    return std::string(taskId);
  }
  const auto second = taskId.find('.', first + 1);
  if (second == std::string_view::npos) {
    return std::string(taskId);
  }
  return std::string(taskId.substr(0, second));
}

std::string_view UcxCompressionCostModel::actionName(Action action) {
  switch (action) {
    case Action::kProbe:
      return "probe";
    case Action::kCompress:
      return "compress";
    case Action::kRaw:
      return "raw";
  }
  return "unknown";
}

} // namespace facebook::velox::ucx_exchange
