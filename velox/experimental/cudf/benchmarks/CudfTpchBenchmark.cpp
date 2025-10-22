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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/Cursor.h"
#include "velox/core/QueryConfig.h"


#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

// Add NVTX support for profiling iterations
#ifdef NVTX_ENABLED
#include <nvtx3/nvtx3.hpp>
#endif

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);

// Declarations needed for the run method override
DECLARE_int32(num_drivers);
DECLARE_int32(num_splits_per_file);
DECLARE_int32(split_preload_per_driver);
DECLARE_int64(preferred_output_batch_bytes);
DECLARE_int32(preferred_output_batch_rows);
DECLARE_int32(max_output_batch_rows);
DECLARE_uint64(max_partial_aggregation_memory);
DECLARE_int32(num_repeats);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override {
    TpchBenchmark::initialize();

    if (FLAGS_velox_cudf_table_scan) {
      connector::unregisterConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName);
      connector::unregisterConnector(
          facebook::velox::exec::test::kHiveConnectorId);

      // Add new values into the cudfHive configuration...
      auto cudfHiveConfigurationValues =
          std::unordered_map<std::string, std::string>();
      cudfHiveConfigurationValues
          [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
              std::to_string(FLAGS_cudf_chunk_read_limit);
      cudfHiveConfigurationValues
          [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
              std::to_string(FLAGS_cudf_pass_read_limit);
      cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                      kAllowMismatchedCudfHiveSchemas] =
          std::to_string(true);
      auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
          std::move(cudfHiveConfigurationValues));

      // Create cudfHive connector with config...
      connector::registerConnectorFactory(
          std::make_shared<
              cudf_velox::connector::hive::CudfHiveConnectorFactory>());
      auto cudfHiveConnector =
          connector::getConnectorFactory(
              facebook::velox::connector::hive::HiveConnectorFactory::
                  kHiveConnectorName)
              ->newConnector(
                  facebook::velox::exec::test::kHiveConnectorId,
                  cudfHiveProperties,
                  ioExecutor_.get());
      connector::registerConnector(cudfHiveConnector);
    }

    // Enable cuDF operators
    cudf_velox::registerCudf();

    // Add custom configs
    queryConfigs_
        [facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
            std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  }

  std::shared_ptr<config::ConfigBase> makeConnectorProperties() override {
    auto cfg = TpchBenchmark::makeConnectorProperties();
    using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

    // CuDF-specific properties.
    cfg->set(
        CudfHiveCfg::kMaxChunkReadLimit,
        std::to_string(FLAGS_cudf_chunk_read_limit));
    cfg->set(
        CudfHiveCfg::kMaxPassReadLimit,
        std::to_string(FLAGS_cudf_pass_read_limit));
    cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

    return cfg;
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const exec::test::TpchPlan& plan) override {
    // TODO (dm): Figure out a way to enforce 1 split per file in
    // CudfHiveDataSource outside of this benchmark
    if (FLAGS_velox_cudf_table_scan) {
      // TODO (dm): Instead of this, we can maybe use
      // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>& filePaths)
      std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
      auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
          path, 1, plan.dataFileFormat);
      for (auto& i : temp) {
        result.push_back(i);
      }
      return result;
    }

    return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
  }

  // Override run method to add NVTX ranges for iteration tracking
  // This implementation mirrors that of QueryBenchmarkBase::run, with annotations for iteration tracking
  std::pair<std::unique_ptr<exec::TaskCursor>, std::vector<RowVectorPtr>> run(
      const exec::test::TpchPlan& tpchPlan,
      const std::unordered_map<std::string, std::string>& queryConfigs = {}) override {
    
    int32_t repeat = 0;
    try {
      for (;;) {
#ifdef NVTX_ENABLED
        // Create NVTX range for each iteration
        std::string rangeName = "TPC-H Iteration " + std::to_string(repeat + 1) + "/" + std::to_string(FLAGS_num_repeats);
        std::cout << "DEBUG: Adding NVTX range: " << rangeName << std::endl;
        nvtx3::scoped_range nvtx_range{rangeName.c_str()};
#else
        std::cout << "DEBUG: NVTX_ENABLED is NOT defined, iteration " << (repeat + 1) << "/" << FLAGS_num_repeats << std::endl;
#endif

        // Execute one iteration (copied from QueryBenchmarkBase::run but without the loop)
        CursorParameters params;
        params.maxDrivers = FLAGS_num_drivers;
        params.planNode = tpchPlan.plan;
        params.queryConfigs = queryConfigs;
        params.queryConfigs[core::QueryConfig::kMaxSplitPreloadPerDriver] =
            std::to_string(FLAGS_split_preload_per_driver);
        params.queryConfigs[core::QueryConfig::kPreferredOutputBatchBytes] =
            std::to_string(FLAGS_preferred_output_batch_bytes);
        params.queryConfigs[core::QueryConfig::kPreferredOutputBatchRows] =
            std::to_string(FLAGS_preferred_output_batch_rows);
        params.queryConfigs[core::QueryConfig::kMaxOutputBatchRows] =
            std::to_string(FLAGS_max_output_batch_rows);
        params.queryConfigs[core::QueryConfig::kMaxPartialAggregationMemory] =
            std::to_string(FLAGS_max_partial_aggregation_memory);
        const int numSplitsPerFile = FLAGS_num_splits_per_file;

        auto addSplits = [&](TaskCursor* taskCursor) {
          auto& task = taskCursor->task();
          if (!taskCursor->noMoreSplits()) {
            for (const auto& entry : tpchPlan.dataFiles) {
              for (const auto& path : entry.second) {
                auto splits = listSplits(path, numSplitsPerFile, tpchPlan);
                for (auto split : splits) {
                  task->addSplit(entry.first, exec::Split(std::move(split)));
                }
              }
              task->noMoreSplits(entry.first);
            }
          }
          taskCursor->setNoMoreSplits();
        };
        auto result = readCursor(params, addSplits);
        ensureTaskCompletion(result.first->task().get());

#ifdef NVTX_ENABLED
        std::cout << "DEBUG: NVTX range for iteration " << (repeat + 1) << " terminated" << std::endl;
#endif

        if (++repeat >= FLAGS_num_repeats) {
          return result;
        }
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Query terminated with: " << e.what();
      return {nullptr, std::vector<RowVectorPtr>()};
    }
  }

  void shutdown() override {
    cudf_velox::unregisterCudf();
    TpchBenchmark::shutdown();
  }
};

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  
  // Create scope guard for guaranteed buffer flush on benchmark completion/exception
  cudf_velox::CallSiteFlushGuard flush_guard;
  
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
