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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <folly/init/Init.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace facebook::velox;

// Multi-threaded reproducer that mimics the 8-driver benchmark behavior
// Uses the same cuDF API as the existing reproducer but with multiple threads
// to trigger memory allocator race conditions

std::atomic<int> successCount{0};
std::atomic<int> errorCount{0};

void workerThread(int threadId, const std::string& filePath, int iterations) {
    try {
        std::cout << "Thread " << threadId << " starting..." << std::endl;
        
        for (int i = 0; i < iterations; ++i) {
            // Use the EXACT same cuDF API as the existing reproducer
            auto readerOptions =
                cudf::io::parquet_reader_options::builder(cudf::io::source_info{filePath})
                    .skip_rows(0)
                    .use_pandas_metadata(true)
                    .use_arrow_schema(true)
                    .allow_mismatched_pq_schemas(false)
                    .build();
            
            // Get stream exactly like the existing reproducer
            auto stream = facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
            
            // Create chunked reader - this is where memory allocation happens
            auto reader = cudf::io::chunked_parquet_reader(
                0, // chunkLimit = 0 (no limit)
                0, // passLimit = 0 (no limit)
                readerOptions,
                stream,
                cudf::get_current_device_resource_ref() // This uses the configured memory resource
            );
            
            int chunkCount = 0;
            size_t totalRows = 0;
            
            // Read chunks - this is where the race condition should occur
            while (reader.has_next()) {
                auto [table, metadata] = reader.read_chunk(); // EXACT call from existing reproducer
                
                chunkCount++;
                totalRows += table->num_rows();
                
                // Small delay to increase chance of race condition
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            
            std::cout << "Thread " << threadId << " iteration " << i << " - Success: " 
                     << chunkCount << " chunks, " << totalRows << " rows" << std::endl;
            successCount++;
        }
        
        std::cout << "Thread " << threadId << " completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Thread " << threadId << " FAILED: " << e.what() << std::endl;
        errorCount++;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <parquet_file> <num_threads> <iterations_per_thread>" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem/lineitem.parquet 8 5" << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "This reproducer tests for memory allocator race conditions in cuDF." << std::endl;
        std::cerr << "Set VELOX_CUDF_MEMORY_RESOURCE environment variable to test different allocators:" << std::endl;
        std::cerr << "  cuda (should work), pool (should fail), async (may fail)" << std::endl;
        return 1;
    }

    std::string parquetFile = argv[1];
    int numThreads = std::stoi(argv[2]);
    int iterationsPerThread = std::stoi(argv[3]);

    std::cout << "=== Multi-threaded cuDF Memory Race Reproducer ===" << std::endl;
    std::cout << "Parquet file: " << parquetFile << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Iterations per thread: " << iterationsPerThread << std::endl;
    std::cout << "Total operations: " << (numThreads * iterationsPerThread) << std::endl;
    
    // Show current memory resource
    const char* memResource = std::getenv("VELOX_CUDF_MEMORY_RESOURCE");
    std::cout << "Memory resource: " << (memResource ? memResource : "default (async)") << std::endl;
    std::cout << "" << std::endl;
    
    try {
        // Initialize Velox (same as existing reproducer)
        folly::Init init{&argc, &argv, false};
        
        // Register cuDF operators (same as existing reproducer)
        cudf_velox::registerCudf();
        
        std::cout << "Starting " << numThreads << " concurrent threads..." << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Launch worker threads (simulating multiple drivers)
        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(workerThread, i, parquetFile, iterationsPerThread);
        }
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "" << std::endl;
        std::cout << "=== Results ===" << std::endl;
        std::cout << "Total time: " << duration.count() << "ms" << std::endl;
        std::cout << "Successful operations: " << successCount.load() << std::endl;
        std::cout << "Failed operations: " << errorCount.load() << std::endl;
        
        if (errorCount.load() > 0) {
            std::cout << "" << std::endl;
            std::cout << "*** RACE CONDITION DETECTED ***" << std::endl;
            std::cout << "This confirms the cuDF memory allocator has thread safety issues!" << std::endl;
            std::cout << "Try with VELOX_CUDF_MEMORY_RESOURCE=cuda to see if it resolves the issue." << std::endl;
        } else {
            std::cout << "All operations completed successfully - no race condition detected with current memory resource" << std::endl;
        }
        
        // Clean up (same as existing reproducer)
        cudf_velox::unregisterCudf();
        
        return errorCount.load() > 0 ? 1 : 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}