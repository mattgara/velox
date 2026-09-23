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

#include <cudf/contiguous_split.hpp>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "velox/vector/TypeAliases.h"

namespace facebook::velox::ucx_exchange {

/// @brief Key for identifying intra-node transfer entries in the registry.
/// Used when UcxExchangeServer and UcxExchangeSource are on the same node.
struct IntraNodeTransferKey {
  std::string taskId;
  uint32_t destination;
  uint32_t sequenceNumber;

  bool operator<(const IntraNodeTransferKey& other) const {
    if (taskId != other.taskId) {
      return taskId < other.taskId;
    }
    if (destination != other.destination) {
      return destination < other.destination;
    }
    return sequenceNumber < other.sequenceNumber;
  }
};

/// @brief Result from intra-node transfer containing data and end marker.
struct IntraNodeTransferResult {
  std::shared_ptr<cudf::packed_columns> data;
  /// Logical rows in 'data'. Carried explicitly for the same reason the remote
  /// path puts a row count on the wire: a packed table with no columns cannot
  /// report its own row count.
  vector_size_t numRows{0};
  bool atEnd{false}; // True if this is the end-of-stream marker
};

/// @brief Entry in the intra-node transfer registry containing the data and
/// one-shot notifications. The server publishes data via publish(), while the
/// source registers a notification and takes the data via poll().
using IntraNodeTransferCallback = std::function<void()>;

struct IntraNodeTransferEntry {
  std::shared_ptr<cudf::packed_columns> data;
  vector_size_t numRows{
      0}; // Logical rows in 'data'; see IntraNodeTransferResult.
  bool atEnd{false}; // True if this is the end-of-stream marker
  IntraNodeTransferCallback onReady;
  IntraNodeTransferCallback onRetrieved;
  bool ready{false}; // True when data is ready to retrieve
};

/// @brief Singleton registry for intra-node data transfers.
///
/// When UcxExchangeServer and UcxExchangeSource are on the same node
/// (same Communicator instance), this registry enables direct data sharing
/// without going through UCXX network transfers.
///
/// The server publishes packed_columns data to the registry, and the source
/// waits for a one-shot ready notification before retrieving it. The producer
/// receives a one-shot notification after retrieval. Callbacks are always
/// invoked outside registry locks.
class IntraNodeTransferRegistry {
 public:
  /// @brief Get the singleton instance of the registry.
  static std::shared_ptr<IntraNodeTransferRegistry> getInstance();

  // Prevent copying
  IntraNodeTransferRegistry(const IntraNodeTransferRegistry&) = delete;
  IntraNodeTransferRegistry& operator=(const IntraNodeTransferRegistry&) =
      delete;

  /// @brief Publish data and notify a waiting source.
  /// The producer must synchronize its CUDA stream before calling publish()
  /// so that the GPU data is ready when the consumer reads it.
  /// @param key The unique key identifying this transfer (taskId, dest, seq)
  /// @param data The packed_columns data to share (nullptr for atEnd)
  /// @param numRows Logical rows in 'data'; 0 for atEnd
  /// @param atEnd True if this is the end-of-stream marker
  /// @param onRetrieved Invoked once after the source retrieves the data, or
  ///        immediately if the producing task has already been cancelled
  void publish(
      const IntraNodeTransferKey& key,
      std::shared_ptr<cudf::packed_columns> data,
      vector_size_t numRows,
      bool atEnd,
      IntraNodeTransferCallback onRetrieved);

  /// @brief Registers a one-shot notification for data becoming ready.
  /// Invokes the callback immediately when data is already available or the
  /// task has been cancelled.
  void notifyWhenReady(
      const IntraNodeTransferKey& key,
      IntraNodeTransferCallback onReady);

  /// @brief Non-blocking poll for intra-node transfer data.
  /// Returns immediately whether data is available or not.
  /// @param key The unique key identifying this transfer
  /// @return nullopt if not ready, or IntraNodeTransferResult if ready.
  ///         Data is nullptr when atEnd is true.
  [[nodiscard]] std::optional<IntraNodeTransferResult> poll(
      const IntraNodeTransferKey& key);

  /// @brief Cancel all pending transfers for a task.
  /// Called when a producing task is removed. Subsequent poll() calls for
  /// this taskId will return an atEnd result instead of nullopt.
  /// @param taskId The task to cancel
  void cancelTask(std::string_view taskId);

  /// @brief Remove a task from the cancelled set.
  /// Called when a task ID is (re)initialized, so that the cancelledTasks_
  /// set does not grow unboundedly across queries.
  /// @param taskId The task to clear from the cancelled set
  void clearCancelledTask(std::string_view taskId);

 private:
  IntraNodeTransferRegistry() = default;

  std::map<IntraNodeTransferKey, std::shared_ptr<IntraNodeTransferEntry>>
      registry_;
  std::unordered_set<std::string> cancelledTasks_;
  std::mutex mutex_;
};

} // namespace facebook::velox::ucx_exchange
