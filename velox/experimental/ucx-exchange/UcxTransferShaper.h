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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace facebook::velox::ucx_exchange {

/// Delays CUDA-IPC send completions to model one aggregate link direction per
/// worker. The actual payload remains in GPU memory and is still transferred
/// by UCX over CUDA IPC.
class UcxTransferShaper {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  using Callback = std::function<void()>;

  UcxTransferShaper(
      double gbytesPerSecond,
      std::chrono::microseconds latency);
  ~UcxTransferShaper();

  UcxTransferShaper(const UcxTransferShaper&) = delete;
  UcxTransferShaper& operator=(const UcxTransferShaper&) = delete;

  /// Reserves serialization time for one payload and returns its simulated
  /// completion deadline. Link latency is pipelined and therefore does not
  /// consume serialization capacity for the next payload.
  TimePoint reserve(std::size_t bytes);

  /// Runs callback at or after deadline without blocking the UCX progress
  /// thread.
  void scheduleAt(TimePoint deadline, Callback callback);

 private:
  void run();

  const double bytesPerSecond_;
  const Clock::duration latency_;

  std::mutex mutex_;
  std::condition_variable condition_;
  TimePoint nextSerializationEnd_{Clock::now()};
  std::multimap<TimePoint, Callback> callbacks_;
  bool stopping_{false};
  std::thread thread_;
};

} // namespace facebook::velox::ucx_exchange
