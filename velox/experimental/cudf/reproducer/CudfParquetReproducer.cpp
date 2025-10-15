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
#include <optional>
#include <algorithm>
#include <vector>

using namespace facebook::velox;

// Minimal reproducer that directly uses cuDF to read parquet
// This should trigger the exact unsnap_kernel code path

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options] [parquet_file_path]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --table <name>           TPC-H table to read (customer, lineitem, nation, orders, part, partsupp, region, supplier)" << std::endl;
    std::cout << "  --data-path <path>       Path to TPC-H data directory (default: /data/tpch)" << std::endl;
    std::cout << "  --scale <factor>         Scale factor - read data this many times (default: 1)" << std::endl;
    std::cout << "  --chunk-limit <bytes>    Chunk read limit in bytes (default: 0 = no limit)" << std::endl;
    std::cout << "  --pass-limit <bytes>     Pass read limit in bytes (default: 0 = no limit)" << std::endl;
    std::cout << "  --skip-rows <count>      Number of rows to skip (default: 0)" << std::endl;
    std::cout << "  --max-rows <count>       Maximum rows to read (default: unlimited)" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " --table lineitem --scale 3" << std::endl;
    std::cout << "  " << program << " --table orders --data-path /custom/path --chunk-limit 67108864" << std::endl;
    std::cout << "  " << program << " /path/to/custom.parquet" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string tableName = "";
    std::string dataPath = "/data/tpch";
    std::string customFile = "";
    int scaleFactor = 1;
    size_t chunkLimit = 0;
    size_t passLimit = 0;
    int64_t skipRows = 0;
    std::optional<cudf::size_type> maxRows = std::nullopt;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--table" && i + 1 < argc) {
            tableName = argv[++i];
        } else if (arg == "--data-path" && i + 1 < argc) {
            dataPath = argv[++i];
        } else if (arg == "--scale" && i + 1 < argc) {
            scaleFactor = std::stoi(argv[++i]);
        } else if (arg == "--chunk-limit" && i + 1 < argc) {
            chunkLimit = std::stoull(argv[++i]);
        } else if (arg == "--pass-limit" && i + 1 < argc) {
            passLimit = std::stoull(argv[++i]);
        } else if (arg == "--skip-rows" && i + 1 < argc) {
            skipRows = std::stoll(argv[++i]);
        } else if (arg == "--max-rows" && i + 1 < argc) {
            maxRows = static_cast<cudf::size_type>(std::stoll(argv[++i]));
        } else if (arg.substr(0, 2) != "--") {
            customFile = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Determine file path
    std::string filePath;
    if (!customFile.empty()) {
        filePath = customFile;
    } else if (!tableName.empty()) {
        // Validate table name
        std::vector<std::string> validTables = {"customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"};
        if (std::find(validTables.begin(), validTables.end(), tableName) == validTables.end()) {
            std::cerr << "Invalid table name: " << tableName << std::endl;
            std::cerr << "Valid tables: customer, lineitem, nation, orders, part, partsupp, region, supplier" << std::endl;
            return 1;
        }
        filePath = dataPath + "/" + tableName + "/" + tableName + ".parquet";
    } else {
        // Default to lineitem
        filePath = dataPath + "/lineitem/lineitem.parquet";
    }
    
    // Validate scale factor
    if (scaleFactor < 1) {
        std::cerr << "Scale factor must be >= 1" << std::endl;
        return 1;
    }

    std::cout << "=== Minimal Velox cuDF Parquet Reproducer ===" << std::endl;
    std::cout << "Parquet file: " << filePath << std::endl;
    std::cout << "Scale factor: " << scaleFactor << " (will read " << scaleFactor << " time(s))" << std::endl;
    std::cout << "Chunk limit: " << (chunkLimit == 0 ? "unlimited" : std::to_string(chunkLimit) + " bytes") << std::endl;
    std::cout << "Pass limit: " << (passLimit == 0 ? "unlimited" : std::to_string(passLimit) + " bytes") << std::endl;
    std::cout << "Skip rows: " << skipRows << std::endl;
    std::cout << "Max rows: " << (maxRows.has_value() ? std::to_string(maxRows.value()) : "unlimited") << std::endl;
    
    try {
        // Initialize Velox
        folly::Init init{&argc, &argv, false};
        
        // Register cuDF operators
        cudf_velox::registerCudf();
        
        std::cout << "Reading parquet file with cuDF chunked reader..." << std::endl;
        
        // Use the EXACT same cuDF API as Velox ParquetDataSource (lines 310-318)
        auto readerOptions =
            cudf::io::parquet_reader_options::builder(cudf::io::source_info{filePath})
                .skip_rows(skipRows)                    // Configurable skip rows
                .use_pandas_metadata(true)              // ParquetConfig default  
                .use_arrow_schema(true)                 // ParquetConfig default
                .allow_mismatched_pq_schemas(false)     // ParquetConfig default
                .build();
        
        // Set max rows if specified
        if (maxRows.has_value()) {
            readerOptions.set_num_rows(maxRows.value());
        }
        
        // Get stream exactly like ParquetDataSource (line 356)
        auto stream = facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
        
        // TEST: Force memory initialization by creating a pool memory resource
        // that zeros out allocated memory (if available)
        std::cout << "Using memory resource: default" << std::endl;
        
        std::cout << "Reading parquet file with cuDF chunked reader..." << std::endl;
        
        int totalChunkCount = 0;
        size_t totalRows = 0;
        
        // Read the file scaleFactor times to simulate larger workload
        for (int scale = 1; scale <= scaleFactor; scale++) {
            std::cout << "=== Scale iteration " << scale << "/" << scaleFactor << " ===" << std::endl;
            
            // Create a new chunked reader for each scale iteration
            auto scaleReader = cudf::io::chunked_parquet_reader(
                chunkLimit,
                passLimit,
                readerOptions,
                stream,
                cudf::get_current_device_resource_ref()
            );
            
            int scaleChunkCount = 0;
            size_t scaleRows = 0;
            
            // Read chunks exactly like Velox ParquetDataSource does
            while (scaleReader.has_next()) {
                auto [table, metadata] = scaleReader.read_chunk(); // This is the EXACT call from line 156!
                
                scaleChunkCount++;
                totalChunkCount++;
                scaleRows += table->num_rows();
                totalRows += table->num_rows();
                
                std::cout << "Scale " << scale << " - Chunk " << scaleChunkCount << ": " << table->num_rows() << " rows, " 
                          << table->num_columns() << " columns" << std::endl;
            }
            
            std::cout << "Scale " << scale << " completed: " << scaleChunkCount << " chunks, " << scaleRows << " rows" << std::endl;
        }
        
        std::cout << "Successfully read parquet data!" << std::endl;
        std::cout << "Total scale iterations: " << scaleFactor << std::endl;
        std::cout << "Total chunks: " << totalChunkCount << std::endl;
        std::cout << "Total rows: " << totalRows << std::endl;
        
        // Clean up
        cudf_velox::unregisterCudf();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
