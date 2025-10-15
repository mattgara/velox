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
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>

using namespace facebook::velox;

// Multi-threaded reproducer that uses the same cuDF API as the existing reproducer
// but with multiple threads to trigger memory allocator race conditions

std::atomic<int> successCount{0};
std::atomic<int> errorCount{0};

void workerThread(int threadId, const std::vector<std::string>& filePaths, int iterations, size_t chunkLimit) {
    try {
        std::cout << "Thread " << threadId << " starting with " << filePaths.size() << " files..." << std::endl;
        
        for (int i = 0; i < iterations; ++i) {
            // Each thread processes different files (simulating different drivers/splits)
            for (size_t fileIdx = 0; fileIdx < filePaths.size(); ++fileIdx) {
                const auto& filePath = filePaths[fileIdx];
                
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
                
                // Use configurable chunk limit
                auto reader = cudf::io::chunked_parquet_reader(
                    chunkLimit, // Limit chunk size to avoid memory exhaustion
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
                    
                    // Force table to go out of scope to free memory
                    table.reset();
                    
                    // Small delay to increase chance of race condition during allocation/deallocation
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
                
                std::cout << "Thread " << threadId << " iteration " << i << " file " << fileIdx 
                         << " - Success: " << chunkCount << " chunks, " << totalRows << " rows" << std::endl;
                successCount++;
            }
        }
        
        std::cout << "Thread " << threadId << " completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Thread " << threadId << " FAILED: " << e.what() << std::endl;
        errorCount++;
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <parquet_directory_or_file> <num_threads> <iterations_per_thread> [max_files] [chunk_limit_mb] [memory_percent]" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5 4    # Limit to 4 files" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5 4 64    # 64MB chunks" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5 4 64 80    # 64MB chunks, 80% GPU memory" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem/lineitem.parquet 8 5 1 1024 90    # 1GB chunks, 90% GPU memory" << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "This reproducer tests for memory allocator race conditions in cuDF." << std::endl;
        std::cerr << "Uses the same cuDF API as the existing reproducer but with multiple threads." << std::endl;
        std::cerr << "Parameters:" << std::endl;
        std::cerr << "  max_files: Limit number of files processed (useful to avoid memory exhaustion)" << std::endl;
        std::cerr << "  chunk_limit_mb: Chunk size in MB (default: 1024MB like benchmark, try 64MB to avoid pool limits)" << std::endl;
        std::cerr << "  memory_percent: Percentage of GPU memory for RMM pool (default: 80%, try 90% for more memory)" << std::endl;
        std::cerr << "Set VELOX_CUDF_MEMORY_RESOURCE environment variable to test different allocators:" << std::endl;
        std::cerr << "  cuda (should work), pool (should fail), async (may fail)" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    int numThreads = std::stoi(argv[2]);
    int iterationsPerThread = std::stoi(argv[3]);
    int maxFiles = (argc >= 5) ? std::stoi(argv[4]) : -1; // -1 means no limit
    int chunkLimitMB = (argc >= 6) ? std::stoi(argv[5]) : 1024; // Default 1024MB like benchmark
    int memoryPercent = (argc == 7) ? std::stoi(argv[6]) : 80; // Default 80% of GPU memory
    
    size_t chunkLimit = static_cast<size_t>(chunkLimitMB) * 1024 * 1024; // Convert MB to bytes

    // Discover parquet files (like the benchmark does)
    std::vector<std::string> parquetFiles;
    
    // Check if input is a directory or file
    struct stat pathStat;
    if (stat(inputPath.c_str(), &pathStat) != 0) {
        std::cerr << "ERROR: Path does not exist: " << inputPath << std::endl;
        return 1;
    }
    
    if (S_ISDIR(pathStat.st_mode)) {
        // Directory - find all parquet files
        std::cout << "Discovering parquet files in directory: " << inputPath << std::endl;
        
        // Simple directory scan for .parquet files
        DIR* dir = opendir(inputPath.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.size() > 8 && filename.substr(filename.size() - 8) == ".parquet") {
                    parquetFiles.push_back(inputPath + "/" + filename);
                }
            }
            closedir(dir);
        }
        
        if (parquetFiles.empty()) {
            std::cerr << "ERROR: No .parquet files found in directory: " << inputPath << std::endl;
            return 1;
        }
        
        std::sort(parquetFiles.begin(), parquetFiles.end());
        std::cout << "Found " << parquetFiles.size() << " parquet files" << std::endl;
        
        // Limit number of files if requested
        if (maxFiles > 0 && parquetFiles.size() > static_cast<size_t>(maxFiles)) {
            parquetFiles.resize(maxFiles);
            std::cout << "Limited to " << maxFiles << " files to avoid memory exhaustion" << std::endl;
        }
    } else {
        // Single file
        parquetFiles.push_back(inputPath);
        std::cout << "Using single parquet file: " << inputPath << std::endl;
    }

    std::cout << "=== Multi-threaded cuDF Memory Race Reproducer ===" << std::endl;
    std::cout << "Parquet files: " << parquetFiles.size() << std::endl;
    for (size_t i = 0; i < parquetFiles.size(); ++i) {
        std::cout << "  [" << i << "] " << parquetFiles[i] << std::endl;
    }
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Iterations per thread: " << iterationsPerThread << std::endl;
    std::cout << "Chunk limit: " << chunkLimitMB << "MB (" << chunkLimit << " bytes)" << std::endl;
    std::cout << "Memory percent: " << memoryPercent << "% of GPU memory" << std::endl;
    std::cout << "Total operations: " << (numThreads * iterationsPerThread * parquetFiles.size()) << std::endl;
    
    // Show current memory resource
    const char* memResource = std::getenv("VELOX_CUDF_MEMORY_RESOURCE");
    std::cout << "Memory resource: " << (memResource ? memResource : "default (async)") << std::endl;
    std::cout << "" << std::endl;
    
    try {
        // Initialize Velox (same as existing reproducer)
        folly::Init init{&argc, &argv, false};
        
        // Register cuDF operators with custom memory configuration
        auto cudfOptions = cudf_velox::CudfOptions(false);
        cudfOptions.memoryPercent = memoryPercent; // Use configurable percentage of GPU memory
        cudf_velox::registerCudf(cudfOptions);
        
        std::cout << "Starting " << numThreads << " concurrent threads..." << std::endl;
        
        // Distribute files across threads (like benchmark distributes splits across drivers)
        std::vector<std::vector<std::string>> threadFiles(numThreads);
        for (size_t i = 0; i < parquetFiles.size(); ++i) {
            threadFiles[i % numThreads].push_back(parquetFiles[i]);
        }
        
        // Show file distribution
        for (int i = 0; i < numThreads; ++i) {
            std::cout << "Thread " << i << " will process " << threadFiles[i].size() << " files" << std::endl;
        }
        std::cout << "" << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Launch worker threads (simulating multiple drivers)
        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(workerThread, i, threadFiles[i], iterationsPerThread, chunkLimit);
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