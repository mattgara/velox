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
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include <glog/logging.h>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {

/* static */
std::shared_ptr<IntraNodeTransferRegistry>
IntraNodeTransferRegistry::getInstance() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static std::shared_ptr<IntraNodeTransferRegistry> instance =
      std::shared_ptr<IntraNodeTransferRegistry>(
          new IntraNodeTransferRegistry());
  return instance;
}

void IntraNodeTransferRegistry::publish(
    const IntraNodeTransferKey& key,
    std::shared_ptr<cudf::packed_columns> data,
    vector_size_t numRows,
    bool atEnd,
    IntraNodeTransferCallback onRetrieved) {
  IntraNodeTransferCallback onReady;
  bool entryExisted{false};
  size_t registrySize{0};
  bool cancelled{false};
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // If the task was already cancelled (removeTask was called), don't create
    // a registry entry. Return an already-fulfilled future so the server
    // doesn't block waiting for a source that will never come.
    if (cancelledTasks_.count(key.taskId)) {
      cancelled = true;
    } else {
      auto [it, inserted] = registry_.try_emplace(
          key, std::make_shared<IntraNodeTransferEntry>());
      entryExisted = !inserted;
      auto& entry = it->second;
      VELOX_CHECK(
          !entry->ready,
          "Duplicate intra-node publish for task {}, destination {}, "
          "sequence {}",
          key.taskId,
          key.destination,
          key.sequenceNumber);
      entry->data = std::move(data);
      entry->numRows = numRows;
      entry->atEnd = atEnd;
      entry->onRetrieved = std::move(onRetrieved);
      entry->ready = true;
      if (entry->onReady) {
        onReady = std::move(entry->onReady);
      }
      registrySize = registry_.size();
    }
  }

  if (cancelled) {
    VLOG(2) << "[INTRA-REG] publish skipped (task cancelled): task="
            << key.taskId << " dest=" << key.destination
            << " seq=" << key.sequenceNumber;
    if (onRetrieved) {
      onRetrieved();
    }
    return;
  }

  VLOG(2) << "[INTRA-REG] publish: task=" << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << atEnd << " entryExisted=" << entryExisted
          << " registrySize=" << registrySize;

  if (onReady) {
    onReady();
  }
}

void IntraNodeTransferRegistry::notifyWhenReady(
    const IntraNodeTransferKey& key,
    IntraNodeTransferCallback onReady) {
  bool notifyNow{false};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelledTasks_.count(key.taskId)) {
      notifyNow = true;
    } else {
      auto [it, inserted] = registry_.try_emplace(
          key, std::make_shared<IntraNodeTransferEntry>());
      auto& entry = it->second;
      if (entry->ready) {
        notifyNow = true;
      } else {
        VELOX_CHECK(
            !entry->onReady,
            "Duplicate intra-node ready notification for task {}, "
            "destination {}, sequence {}",
            key.taskId,
            key.destination,
            key.sequenceNumber);
        entry->onReady = std::move(onReady);
      }
    }
  }

  if (notifyNow && onReady) {
    onReady();
  }
}

std::optional<IntraNodeTransferResult> IntraNodeTransferRegistry::poll(
    const IntraNodeTransferKey& key) {
  IntraNodeTransferCallback onRetrieved;
  IntraNodeTransferResult result;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if this task has been cancelled (producer removed).
    if (cancelledTasks_.count(key.taskId)) {
      VLOG(2) << "[INTRA-REG] poll cancelled: task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber;
      return IntraNodeTransferResult{
          .data = nullptr, .numRows = 0, .atEnd = true};
    }

    auto it = registry_.find(key);
    if (it == registry_.end()) {
      // No entry yet - server hasn't published
      VLOG(3) << "[INTRA-REG] poll miss (no entry): task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber
              << " registrySize=" << registry_.size();
      return std::nullopt;
    }
    auto& entry = it->second;
    if (!entry->ready) {
      // Entry exists but data not ready yet
      VLOG(3) << "[INTRA-REG] poll miss (not ready): task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber;
      return std::nullopt;
    }

    result.data = std::move(entry->data);
    result.numRows = entry->numRows;
    result.atEnd = entry->atEnd;
    onRetrieved = std::move(entry->onRetrieved);
    registry_.erase(it);
  }

  if (onRetrieved) {
    onRetrieved();
  }

  VLOG(2) << "[INTRA-REG] poll hit: task=" << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << result.atEnd;

  return result;
}

void IntraNodeTransferRegistry::cancelTask(std::string_view taskId) {
  std::vector<IntraNodeTransferCallback> callbacks;
  size_t entriesCleaned{0};

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelledTasks_.insert(std::string{taskId});

    // Wake both sides outside the registry lock. Sources will observe the
    // cancelled task as an end marker, and producers can finish immediately.
    for (auto it = registry_.begin(); it != registry_.end();) {
      if (it->first.taskId == taskId) {
        if (it->second->onReady) {
          callbacks.push_back(std::move(it->second->onReady));
        }
        if (it->second->onRetrieved) {
          callbacks.push_back(std::move(it->second->onRetrieved));
        }
        ++entriesCleaned;
        it = registry_.erase(it);
      } else {
        ++it;
      }
    }
  }

  for (auto& callback : callbacks) {
    callback();
  }

  VLOG(2) << "[INTRA-REG] cancelTask: task=" << taskId
          << " entriesCleaned=" << entriesCleaned;
}

void IntraNodeTransferRegistry::clearCancelledTask(std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  cancelledTasks_.erase(std::string{taskId});
}

} // namespace facebook::velox::ucx_exchange
