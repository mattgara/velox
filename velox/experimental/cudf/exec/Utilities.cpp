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

#include <common/base/Exceptions.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <cstdint>     // For uint64_t, uintptr_t
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <algorithm>  // For std::sort

#include <cuda_runtime.h>
#include <execinfo.h>  // For backtrace
#include <csignal>     // For signal handling

namespace facebook::velox::cudf_velox {

// Thread-local storage for stack trace entries
struct StackTraceEntry {
  std::string timestamp;
  uintptr_t pointer;
  size_t size;
  uint64_t stream;
  std::string stack_trace;
};

// Global collection of all thread-local buffers
static std::vector<std::vector<StackTraceEntry>*> g_thread_buffers;
static std::mutex g_buffer_registry_mutex;

// Thread-local buffer for this thread's stack traces
thread_local std::vector<StackTraceEntry> t_stack_trace_buffer;
thread_local bool t_buffer_registered = false;

// Signal handler registration state
static bool g_signal_handlers_installed = false;
static std::mutex g_signal_handler_mutex;

// Forward declaration
void flushStackTraceBuffers();

// Signal handler for crash-safe stack trace flushing
void crashSignalHandler(int signal) {
  // Try to flush stack traces before crashing
  try {
    flushStackTraceBuffers();
  } catch (...) {
    // Ignore any errors during crash handling
  }
  
  // Re-raise the signal to get normal crash behavior
  std::signal(signal, SIG_DFL);
  std::raise(signal);
}

// Install signal handlers for crash-safe flushing
void installCrashHandlers() {
  std::lock_guard<std::mutex> lock(g_signal_handler_mutex);
  if (g_signal_handlers_installed) return;
  
  // Install handlers for common crash signals
  std::signal(SIGSEGV, crashSignalHandler);  // Segmentation fault
  std::signal(SIGABRT, crashSignalHandler);  // Abort
  std::signal(SIGFPE, crashSignalHandler);   // Floating point exception
  std::signal(SIGILL, crashSignalHandler);   // Illegal instruction
  std::signal(SIGBUS, crashSignalHandler);   // Bus error
  
  g_signal_handlers_installed = true;
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

[[nodiscard]] auto makeArenaMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

[[nodiscard]] auto makeManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(percent));
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
  else {
    VELOX_FAIL(
        "Unknown memory resource mode: " + std::string(mode) +
        "\nExpecting: cuda, pool, async, arena, managed, or managed_pool");
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
        // DEBUG: Synchronize device before deallocate to catch use-after-free timing issues
        cudaDeviceSynchronize();
        
        // Capture stack trace for deallocate calls
        captureStackTrace(ptr, bytes, stream);
        
        logging_mr_.deallocate(ptr, bytes, stream);
      }
      
    private:
      void captureStackTrace(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) {
        const char* stack_trace_file = std::getenv("RMM_STACK_TRACE_FILE");
        if (!stack_trace_file) return;
        
        // Install crash handlers on first use
        installCrashHandlers();
        
        // Register this thread's buffer if not already done
        if (!t_buffer_registered) {
          std::lock_guard<std::mutex> lock(g_buffer_registry_mutex);
          g_thread_buffers.push_back(&t_stack_trace_buffer);
          t_buffer_registered = true;
        }
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        // Capture stack trace (use large buffer for deep call stacks)
        void* trace[512];
        int trace_size = backtrace(trace, 512);
        char** symbols = backtrace_symbols(trace, trace_size);
        
        // Build stack trace string (escape commas and newlines for CSV)
        std::stringstream stack_trace_ss;
        for (int i = 0; i < trace_size; ++i) {
          std::string symbol = symbols[i] ? symbols[i] : "unknown";
          // Replace commas and newlines to keep CSV format clean
          for (char& c : symbol) {
            if (c == ',' || c == '\n' || c == '\r') c = '|';
          }
          if (i > 0) stack_trace_ss << ";";
          stack_trace_ss << symbol;
        }
        
        // Build timestamp string
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%H:%M:%S") 
                     << "." << std::setfill('0') << std::setw(3) << ms.count();
        
        // Store in thread-local buffer (no mutex needed - thread-local)
        StackTraceEntry entry;
        entry.timestamp = timestamp_ss.str();
        entry.pointer = reinterpret_cast<uintptr_t>(ptr);
        entry.size = bytes;
        entry.stream = stream.value();
        entry.stack_trace = stack_trace_ss.str();
        
        t_stack_trace_buffer.push_back(std::move(entry));
        
        free(symbols);
      }
      
      bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override {
        return logging_mr_.is_equal(other);
      }
    };
    
    return std::make_shared<LoggingWrapper>(mr, logPath);
  }
  
  return mr;
}

// Function to flush all thread-local stack trace buffers to file
void flushStackTraceBuffers() {
  const char* stack_trace_file = std::getenv("RMM_STACK_TRACE_FILE");
  if (!stack_trace_file) return;
  
  std::lock_guard<std::mutex> lock(g_buffer_registry_mutex);
  
  std::ofstream csv_file(stack_trace_file);
  if (!csv_file.is_open()) return;
  
  // Write header
  csv_file << "Timestamp,Pointer,Size,Stream,StackTrace\n";
  
  // Collect all entries from all thread buffers
  std::vector<StackTraceEntry> all_entries;
  for (auto* buffer : g_thread_buffers) {
    if (buffer) {
      all_entries.insert(all_entries.end(), buffer->begin(), buffer->end());
    }
  }
  
  // Sort by timestamp for chronological order
  std::sort(all_entries.begin(), all_entries.end(), 
    [](const StackTraceEntry& a, const StackTraceEntry& b) {
      return a.timestamp < b.timestamp;
    });
  
  // Write all entries
  for (const auto& entry : all_entries) {
    csv_file << entry.timestamp
             << ",0x" << std::hex << entry.pointer
             << "," << std::dec << entry.size
             << "," << entry.stream
             << ",\"" << entry.stack_trace << "\"\n";
  }
  
  csv_file.close();
  
  // Clear all buffers
  for (auto* buffer : g_thread_buffers) {
    if (buffer) buffer->clear();
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

} // namespace facebook::velox::cudf_velox
