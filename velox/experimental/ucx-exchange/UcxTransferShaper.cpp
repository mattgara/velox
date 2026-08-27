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

#include <algorithm>
#include <utility>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {

UcxTransferShaper::UcxTransferShaper(
    double gbytesPerSecond,
    std::chrono::microseconds latency)
    : bytesPerSecond_(gbytesPerSecond * 1e9),
      latency_(std::chrono::duration_cast<Clock::duration>(latency)) {
  VELOX_CHECK_GT(gbytesPerSecond, 0.0);
  thread_ = std::thread([this] { run(); });
}

UcxTransferShaper::~UcxTransferShaper() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  condition_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

UcxTransferShaper::TimePoint UcxTransferShaper::reserve(std::size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto start = std::max(Clock::now(), nextSerializationEnd_);
  const auto serialization = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(bytes / bytesPerSecond_));
  nextSerializationEnd_ = start + serialization;
  return nextSerializationEnd_ + latency_;
}

void UcxTransferShaper::scheduleAt(TimePoint deadline, Callback callback) {
  bool runImmediately = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      runImmediately = true;
    } else {
      callbacks_.emplace(deadline, std::move(callback));
    }
  }
  if (runImmediately) {
    callback();
    return;
  }
  condition_.notify_one();
}

void UcxTransferShaper::run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopping_) {
    if (callbacks_.empty()) {
      condition_.wait(lock, [this] { return stopping_ || !callbacks_.empty(); });
      continue;
    }

    const auto deadline = callbacks_.begin()->first;
    condition_.wait_until(lock, deadline, [this, deadline] {
      return stopping_ ||
          (!callbacks_.empty() && callbacks_.begin()->first < deadline);
    });
    if (stopping_ || callbacks_.empty() ||
        Clock::now() < callbacks_.begin()->first) {
      continue;
    }

    auto callback = std::move(callbacks_.begin()->second);
    callbacks_.erase(callbacks_.begin());
    lock.unlock();
    callback();
    lock.lock();
  }
  callbacks_.clear();
}

} // namespace facebook::velox::ucx_exchange
