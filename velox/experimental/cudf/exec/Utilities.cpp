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

#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/prefetch_resource_adaptor.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <common/base/Exceptions.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <vector>
#include <unordered_set>

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <execinfo.h>

namespace facebook::velox::cudf_velox {

// Simple direct-to-file logging
static std::mutex g_csv_file_mutex;

// Structure to hold call site information
struct CallSiteInfo {
  std::string module_name;
  uintptr_t module_base;
  uintptr_t call_offset;
  std::vector<uintptr_t> stack_offsets;  // Multi-level stack trace
};

// Extract call site information with multi-level stack trace
CallSiteInfo getCallSiteInfo() {
  CallSiteInfo info;
  info.module_name = "unknown";
  info.module_base = 0;
  info.call_offset = 0;
  
  // Get stack trace depth from environment (default 8)
  static int stack_depth = -1;
  if (stack_depth == -1) {
    const char* depth_env = std::getenv("RMM_STACK_TRACE_DEPTH");
    stack_depth = depth_env ? std::atoi(depth_env) : 8;
    if (stack_depth < 1) stack_depth = 1;  // Minimum 1 level
    if (stack_depth > 32) stack_depth = 32; // Maximum 32 levels for safety
  }
  
  // Capture stack trace using backtrace
  void* return_addrs[32];  // Max possible size
  int stack_size = backtrace(return_addrs, stack_depth + 1); // +1 to account for this function
  
  // Safety checks for backtrace result
  if (stack_size <= 0) return info; // backtrace failed
  if (stack_size < 2) return info;  // Need at least caller frame
  void* primary_addr = return_addrs[1];
  
  // Safety check: ensure primary address is valid
  if (!primary_addr) return info;
  
  info.call_offset = reinterpret_cast<uintptr_t>(primary_addr);
  
  Dl_info dl_info;
  if (dladdr(primary_addr, &dl_info) != 0) {
    if (dl_info.dli_fname) {
      // Extract just the filename, not full path
      const char* filename = strrchr(dl_info.dli_fname, '/');
      info.module_name = filename ? (filename + 1) : dl_info.dli_fname;
    }
    if (dl_info.dli_fbase) {
      info.module_base = reinterpret_cast<uintptr_t>(dl_info.dli_fbase);
      info.call_offset = reinterpret_cast<uintptr_t>(primary_addr) - info.module_base;
    }
  }
  
  // Calculate offsets for all captured stack levels (skip frame 0 = this function)
  for (int i = 1; i < stack_size; i++) {
    void* addr = return_addrs[i];
    
    // Safety checks: skip null pointers and obviously invalid addresses
    if (!addr) break;
    
    // Skip addresses that are clearly invalid (too low or too high)
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
    if (addr_val < 0x1000 || addr_val > 0x7fffffffffff) continue;
    
    Dl_info level_info;
    if (dladdr(addr, &level_info) != 0 && level_info.dli_fbase) {
      uintptr_t level_base = reinterpret_cast<uintptr_t>(level_info.dli_fbase);
      uintptr_t level_offset = addr_val - level_base;
      info.stack_offsets.push_back(level_offset);
    } else {
      // Still include raw address even if dladdr fails (but only if it looks valid)
      info.stack_offsets.push_back(addr_val);
    }
  }
  
  return info;
}

// Simple helper to write one entry directly to CSV file
void writeCallSiteEntry(const char* csv_file, uintptr_t pointer, size_t size, 
                       uint64_t stream, uint64_t thread_id, int device_id,
                       const std::string& module_name, uintptr_t module_base, 
                       uintptr_t call_offset, const std::vector<uintptr_t>& stack_offsets) {
  std::lock_guard<std::mutex> lock(g_csv_file_mutex);
  
  // Open in append mode
  std::ofstream file(csv_file, std::ios::app);
  if (!file.is_open()) return;
  
  // Get timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = now.time_since_epoch().count();
  
  // Build stack trace string
  std::ostringstream stack_trace;
  for (size_t i = 0; i < stack_offsets.size(); i++) {
    if (i > 0) stack_trace << "->";
    stack_trace << "0x" << std::hex << stack_offsets[i];
  }
  
  // Write entry directly to file
  file << ns
       << ",0x" << std::hex << pointer
       << "," << std::dec << size
       << ",0x" << std::hex << stream
       << ",0x" << std::hex << thread_id
       << "," << std::dec << device_id
       << ",\"" << module_name << "\""
       << ",0x" << std::hex << module_base
       << ",0x" << std::hex << call_offset
       << ",\"" << stack_trace.str() << "\"\n";
}

namespace {
[[nodiscard]] auto makeCudaMr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

[[nodiscard]] auto makePoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

[[nodiscard]] auto makeAsyncMr(int percent) {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>(
      rmm::percent_of_free_device_memory(percent));
}

[[nodiscard]] auto makeManagedMr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

/// \brief Makes a prefetched<managed> resource
[[nodiscard]] auto makePrefetchManagedMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedMr());
}

[[nodiscard]] auto makeArenaMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

[[nodiscard]] auto makeManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a prefetched<pool<managed>> resource
[[nodiscard]] auto makePrefetchManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedPoolMr(percent));
}

void enablePrefetching() {
  cudf::experimental::prefetch::enable_prefetching("hash_join");
  cudf::experimental::prefetch::enable_prefetching("gather");
  cudf::experimental::prefetch::enable_prefetching("column_view::get_data");
  cudf::experimental::prefetch::enable_prefetching(
      "mutable_column_view::get_data");
}

} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode,
    int percent) {
  std::shared_ptr<rmm::mr::device_memory_resource> mr;
  
  if (mode == "cuda")
    mr = makeCudaMr();
  else if (mode == "pool")
    mr = makePoolMr(percent);
  else if (mode == "async")
    mr = makeAsyncMr(percent);
  else if (mode == "arena")
    mr = makeArenaMr(percent);
  else if (mode == "managed")
    mr = makeManagedMr();
  else if (mode == "managed_pool")
    mr = makeManagedPoolMr(percent);
  else if (mode == "prefetch_managed") {
    enablePrefetching();
    mr = makePrefetchManagedMr();
  }
  else if (mode == "prefetch_managed_pool") {
    enablePrefetching();
    mr = makePrefetchManagedPoolMr(percent);
  }
  else {
    VELOX_FAIL(
        "Unknown memory resource mode: " + std::string(mode) +
        "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, managed_pool, prefetch_managed_pool");
  }

  // Check if RMM memory event logging is enabled via RMM_LOG_FILE environment variable
  const char* rmm_log_file = std::getenv("RMM_LOG_FILE");
  if (rmm_log_file) {
    std::string logPath(rmm_log_file);
    
    // Wrapper class that holds both resources but acts like the logging resource
    class LoggingWrapper : public rmm::mr::device_memory_resource {
    private:
      std::shared_ptr<rmm::mr::device_memory_resource> upstream_mr_;
      rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource> logging_mr_;
      
    public:
      LoggingWrapper(std::shared_ptr<rmm::mr::device_memory_resource> upstream, const std::string& logFile)
        : upstream_mr_(std::move(upstream)), logging_mr_(upstream_mr_.get(), logFile) {}
      
      // Delegate all device_memory_resource methods to logging_mr_
      void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
        return logging_mr_.allocate(bytes, stream);
      }
      
      void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override {
        // Capture call site for logging (if enabled)
        captureCallSite(ptr, bytes, stream);
        
        logging_mr_.deallocate(ptr, bytes, stream);
      }
      
    private:
      void captureCallSite(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) {
        const char* stack_trace_file = std::getenv("RMM_STACK_TRACE_FILE");
        if (!stack_trace_file) return;
        
        // Get caller address and extract call site info
        CallSiteInfo call_site = getCallSiteInfo();
        
        // Get current device and thread info
        int device_id = -1;
        cudaGetDevice(&device_id);  // Fast call
        uint64_t thread_id = reinterpret_cast<uint64_t>(pthread_self());
        
        // Write directly to file (simple, no buffering)
        writeCallSiteEntry(stack_trace_file, 
                          reinterpret_cast<uintptr_t>(ptr),
                          bytes,
                          reinterpret_cast<uintptr_t>(stream.value()),
                          thread_id,
                          device_id,
                          call_site.module_name,
                          call_site.module_base,
                          call_site.call_offset,
                          call_site.stack_offsets);
      }
      
      bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override {
        return logging_mr_.is_equal(other);
      }
    };
    
    return std::make_shared<LoggingWrapper>(mr, logPath);
  }
  
  return mr;
}

// Simple function to write CSV header once
void flushCallSiteBuffers() {
  const char* stack_trace_file = std::getenv("RMM_STACK_TRACE_FILE");
  if (!stack_trace_file) return;
  
  // Just write the header once - entries are written directly by writeCallSiteEntry
  std::lock_guard<std::mutex> lock(g_csv_file_mutex);
  std::ofstream csv_file(stack_trace_file);
  if (csv_file.is_open()) {
    csv_file << "Timestamp,Pointer,Size,Stream,ThreadID,DeviceID,Module,ModuleBase,CallOffset,StackTrace\n";
  }
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  return cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::table> makeEmptyTable(TypePtr const& inputType) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    if (auto const& childType = inputType->childAt(i);
        childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType);
      auto structColumn = std::make_unique<cudf::column>(
          cudf::data_type(cudf::type_id::STRUCT),
          0,
          rmm::device_buffer(),
          rmm::device_buffer(),
          0,
          tbl->release());
      emptyColumns.push_back(std::move(structColumn));
    } else {
      auto emptyColumn = cudf::make_empty_column(
          cudf_velox::veloxToCudfTypeId(inputType->childAt(i)));
      emptyColumns.push_back(std::move(emptyColumn));
    }
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType);
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  if (tables.size() == 1) {
    return tables[0]->release();
  }

  auto output = cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
  stream.synchronize();
  return output;
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType));
    return concatTables;
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  if (tables.size() == 1) {
    concatTables.push_back(tables[0]->release());
    return concatTables;
  }

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  auto const maxRows =
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
  size_t startpos = 0;
  size_t runningRows = 0;
  for (size_t i = 0; i < tableViews.size(); ++i) {
    auto const numRows = static_cast<size_t>(tableViews[i].num_rows());
    // If adding this table would exceed the limit, flush current batch
    // [startpos, i).
    if (runningRows > 0 && runningRows + numRows > maxRows) {
      outputTables.push_back(cudf::concatenate(
          std::vector<cudf::table_view>(
              tableViews.begin() + startpos, tableViews.begin() + i),
          stream,
          cudf::get_current_device_resource_ref()));
      startpos = i;
      runningRows = 0;
    }
    runningRows += numRows;
  }
  // Flush the final batch [startpos, end).
  if (startpos < tableViews.size()) {
    outputTables.push_back(cudf::concatenate(
        std::vector<cudf::table_view>(
            tableViews.begin() + startpos, tableViews.end()),
        stream,
        cudf::get_current_device_resource_ref()));
  }
  stream.synchronize();
  return outputTables;
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  cudaEventCreateWithFlags(&ev, flags);
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(rmm::cuda_stream_view stream) const {
  cudaEventRecord(event_, stream.value());
  return *this;
}

const CudaEvent& CudaEvent::waitOn(rmm::cuda_stream_view stream) const {
  cudaStreamWaitEvent(stream.value(), event_, 0);
  return *this;
}



} // namespace facebook::velox::cudf_velox
