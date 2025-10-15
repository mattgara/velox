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

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

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
using namespace facebook::velox::cudf_velox::connector::parquet;
using namespace facebook::velox::connector;
using namespace facebook::velox::exec::test;

// Multi-threaded reproducer that mimics the EXACT Velox ParquetDataSource behavior
// This should trigger the exact same memory race condition as the benchmark

std::atomic<int> successCount{0};
std::atomic<int> errorCount{0};

void workerThread(int threadId, const std::vector<std::string>& filePaths, int iterations) {
    try {
        std::cout << "Thread " << threadId << " starting with " << filePaths.size() << " files..." << std::endl;
        
        for (int i = 0; i < iterations; ++i) {
            // Each thread processes different files (simulating different drivers/splits)
            for (size_t fileIdx = 0; fileIdx < filePaths.size(); ++fileIdx) {
                const auto& filePath = filePaths[fileIdx];
                
                // Create the EXACT same schema as lineitem table (from TPC-H)
                auto outputType = ROW({
                    {"l_orderkey", BIGINT()},
                    {"l_partkey", BIGINT()},
                    {"l_suppkey", BIGINT()},
                    {"l_linenumber", INTEGER()},
                    {"l_quantity", DOUBLE()},
                    {"l_extendedprice", DOUBLE()},
                    {"l_discount", DOUBLE()},
                    {"l_tax", DOUBLE()},
                    {"l_returnflag", VARCHAR()},
                    {"l_linestatus", VARCHAR()},
                    {"l_shipdate", DATE()},
                    {"l_commitdate", DATE()},
                    {"l_receiptdate", DATE()},
                    {"l_shipinstruct", VARCHAR()},
                    {"l_shipmode", VARCHAR()},
                    {"l_comment", VARCHAR()}
                });
                
                // Create ParquetConfig with EXACT same settings as benchmark
                auto emptyConfig = std::make_shared<config::ConfigBase>(std::unordered_map<std::string, std::string>{});
                auto parquetConfig = std::make_shared<ParquetConfig>(emptyConfig);
                
                // Create table handle (same as benchmark)
                auto tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
                    "hive_connector",
                    "lineitem",
                    true, // partitioned
                    SubfieldFilters{},
                    nullptr, // remainingFilter
                    nullptr, // dataColumns
                    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>{});
                
                // Create column handles (same as benchmark)
                ColumnHandleMap columnHandles;
                for (int j = 0; j < outputType->size(); ++j) {
                    auto name = outputType->nameOf(j);
                    auto type = outputType->childAt(j);
                    columnHandles[name] = std::make_shared<connector::hive::HiveColumnHandle>(
                        name, connector::hive::HiveColumnHandle::ColumnType::kRegular, type, type);
                }
                
                // Create ParquetDataSource - EXACT same as benchmark
                auto dataSource = std::make_unique<ParquetDataSource>(
                    outputType,
                    tableHandle,
                    columnHandles,
                    nullptr, // executor
                    nullptr, // connectorQueryCtx
                    parquetConfig);
                
                // Create split for the parquet file - EXACT same as benchmark
                auto split = std::make_shared<ParquetConnectorSplit>(
                    "test_connector_id_" + std::to_string(threadId),
                    filePath, // filePath
                    0); // splitWeight
                
                // Add split to data source
                dataSource->addSplit(split);
                
                // This is the EXACT call that triggers the problematic code path
                // Multiple threads doing this concurrently should reproduce the race condition
                ContinueFuture future;
                auto result = dataSource->next(100000, future); // Read up to 100K rows per iteration
                
                if (result.has_value()) {
                    std::cout << "Thread " << threadId << " iteration " << i << " file " << fileIdx 
                             << " - Success: " << result.value()->size() << " rows" << std::endl;
                    successCount++;
                } else {
                    std::cout << "Thread " << threadId << " iteration " << i << " file " << fileIdx 
                             << " - No data" << std::endl;
                }
                
                // Small delay to allow other threads to interleave
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        std::cout << "Thread " << threadId << " completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Thread " << threadId << " FAILED: " << e.what() << std::endl;
        errorCount++;
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <parquet_directory_or_file> <num_threads> <iterations_per_thread> [max_files]" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem 8 5 4    # Limit to 4 files" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/tpch/lineitem/lineitem.parquet 8 5" << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "This reproducer tests for memory allocator race conditions in cuDF." << std::endl;
        std::cerr << "Uses the EXACT same Velox ParquetDataSource code path as the benchmark." << std::endl;
        std::cerr << "Use max_files to limit the number of files processed (useful to avoid memory exhaustion)." << std::endl;
        std::cerr << "Set VELOX_CUDF_MEMORY_RESOURCE environment variable to test different allocators:" << std::endl;
        std::cerr << "  cuda (should work), pool (should fail), async (may fail)" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    int numThreads = std::stoi(argv[2]);
    int iterationsPerThread = std::stoi(argv[3]);
    int maxFiles = (argc == 5) ? std::stoi(argv[4]) : -1; // -1 means no limit

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

    std::cout << "=== Multi-threaded Velox ParquetDataSource Race Reproducer ===" << std::endl;
    std::cout << "Parquet files: " << parquetFiles.size() << std::endl;
    for (size_t i = 0; i < parquetFiles.size(); ++i) {
        std::cout << "  [" << i << "] " << parquetFiles[i] << std::endl;
    }
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Iterations per thread: " << iterationsPerThread << std::endl;
    std::cout << "Total operations: " << (numThreads * iterationsPerThread * parquetFiles.size()) << std::endl;
    
    // Show current memory resource
    const char* memResource = std::getenv("VELOX_CUDF_MEMORY_RESOURCE");
    std::cout << "Memory resource: " << (memResource ? memResource : "default (async)") << std::endl;
    std::cout << "" << std::endl;
    
    try {
        // Initialize Velox (same as benchmark)
        folly::Init init{&argc, &argv, false};
        
        // Register cuDF operators (same as benchmark)
        cudf_velox::registerCudf();
        
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
            threads.emplace_back(workerThread, i, threadFiles[i], iterationsPerThread);
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
        
        // Clean up (same as benchmark)
        cudf_velox::unregisterCudf();
        
        return errorCount.load() > 0 ? 1 : 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}