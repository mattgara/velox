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

#include <cstdint>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfDebugExprTree{"cudf.debug_expr_tree"};
  static constexpr const char* kCudfDebugOperatorFlow{
      "cudf.debug_operator_flow"};
  static constexpr const char* kCudfDebugOperatorFlowSync{
      "cudf.debug_operator_flow_sync"};
  static constexpr const char* kCudfDebugOperatorFlowDeviceSyncPoint{
      "cudf.debug_operator_flow_device_sync_point"};
  static constexpr const char* kCudfDebugHashAggFakeGroupbyMode{
      "cudf.debug_hashagg_fake_groupby_mode"};
  static constexpr const char* kCudfDebugHashAggProbeDumpDir{
      "cudf.debug_hashagg_probe_dump_dir"};
  static constexpr const char* kCudfDebugHashAggProbeDumpMaxRows{
      "cudf.debug_hashagg_probe_dump_max_rows"};
  static constexpr const char* kCudfDebugHashAggDumpDir{
      "cudf.debug_hashagg_dump_dir"};
  static constexpr const char* kCudfDebugHashAggDumpMaxRows{
      "cudf.debug_hashagg_dump_max_rows"};
  static constexpr const char* kCudfDebugHashAggDecimalCpuAggregateMode{
      "cudf.debug_hashagg_decimal_cpu_aggregate_mode"};
  static constexpr const char* kCudfDebugDisableDecimalSumAvgGpu{
      "cudf.debug_disable_decimal_sum_avg_gpu"};
  static constexpr const char* kCudfDebugHashAggEndToEndValidate{
      "cudf.debug_hashagg_end_to_end_validate"};
  static constexpr const char* kCudfDebugHashAggEndToEndMaxRows{
      "cudf.debug_hashagg_end_to_end_max_rows"};
  static constexpr const char* kCudfDebugHashAggEndToEndBatchRows{
      "cudf.debug_hashagg_end_to_end_batch_rows"};
  static constexpr const char* kCudfDebugHashAggExpectedPath{
      "cudf.debug_hashagg_expected_path"};
  static constexpr const char* kCudfDebugHashAggStateRoundtripValidate{
      "cudf.debug_hashagg_state_roundtrip_validate"};
  static constexpr const char* kCudfDebugSerdeValidate{
      "cudf.debug_serde_validate"};
  static constexpr const char* kCudfDebugSerdeValidateMaxRows{
      "cudf.debug_serde_validate_max_rows"};
  static constexpr const char* kCudfDebugCudfToVeloxValidate{
      "cudf.debug_cudf_to_velox_validate"};
  static constexpr const char* kCudfDebugCudfToVeloxMaxRows{
      "cudf.debug_cudf_to_velox_max_rows"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Enable verbose Expr DAG tree logging in CudfFilterProject.
  /// Keep disabled by default because deep recursive traversal can be noisy
  /// and expensive.
  bool debugExprTree{false};

  /// Enable extra operator-level debug logs added for cuDF operator flow
  /// debugging (e.g. CudfTopN/CudfHashAggregation internals).
  bool debugOperatorFlow{false};

  /// Optional: force CUDA stream sync checkpoints in operator-flow debug.
  /// This is expensive and can perturb timing; use only for root-cause
  /// localization.
  bool debugOperatorFlowSync{false};

  /// Optional: inject full device sync probes in cuDF operator flow:
  ///   0  => disabled
  ///  -1  => enable all probe groups
  ///   1  => groupby-core probes
  ///   2  => decimal request/deserialize probes
  ///   3  => decimal reduce/avg probes
  ///   4  => global aggregation probes
  ///   5  => operator-boundary probes
  ///   6  => groupby request-isolation probes
  /// Intended for speculative stream-race triage.
  int32_t debugOperatorFlowDeviceSyncPoint{0};

  /// Optional: bypass hash-aggregate groupby with fake constant output:
  ///   0 => disabled
  ///   1 => output constant zero/default values
  ///   2 => output constant one for primitive numerics where supported
  /// Applied only to final/single groupby steps (partial/intermediate keep
  /// real aggregation to preserve valid intermediate decimal state encoding).
  int32_t debugHashAggFakeGroupbyMode{0};

  /// Optional: when non-empty, request-isolation probe failures dump a
  /// standalone replay bundle under this directory.
  std::string debugHashAggProbeDumpDir;

  /// Optional max input rows allowed for probe bundle dumps:
  ///   0 => no row limit
  ///  >0 => skip dump when row count exceeds this threshold
  int32_t debugHashAggProbeDumpMaxRows{50000};

  /// Optional: when non-empty, dump each groupby input to a replay bundle.
  std::string debugHashAggDumpDir;

  /// Optional max input rows allowed for full groupby dumps:
  ///   0 => no row limit
  ///  >0 => skip dump when row count exceeds this threshold
  int32_t debugHashAggDumpMaxRows{0};

  /// Optional: replace decimal hash groupby aggregate kernel calls with a
  /// host-side emulation path (debug only).
  ///   0 => disabled
  ///   1 => enabled (fails fast if an unsupported aggregate shape is seen)
  int32_t debugHashAggDecimalCpuAggregateMode{0};

  /// Optional: disable cuDF for decimal SUM/AVG and force CPU aggregation.
  /// Default is true to preserve current safety detour; set to false to allow
  /// cuDF decimal SUM/AVG execution.
  bool debugDisableDecimalSumAvgGpu{true};

  /// Optional: end-to-end hash aggregation validation (input->output).
  bool debugHashAggEndToEndValidate{false};

  /// Optional: max input rows to validate (0 = all rows).
  int64_t debugHashAggEndToEndMaxRows{0};

  /// Optional: batch size for GPU->host copies during validation.
  int64_t debugHashAggEndToEndBatchRows{1000000};

  /// Optional: expected SUM results file for HashAgg validation.
  /// Format: key,sum on each line (sum can be decimal string).
  std::string debugHashAggExpectedPath;

  /// Optional: validate decimal sum-state serialization roundtrip.
  bool debugHashAggStateRoundtripValidate{false};

  /// Optional: validate Presto serialization roundtrip for RowVector outputs.
  bool debugSerdeValidate{false};

  /// Optional: max rows to validate per batch (0 = all rows).
  int64_t debugSerdeValidateMaxRows{100000};

  /// Optional: validate cuDF->Velox conversion by comparing column values.
  bool debugCudfToVeloxValidate{false};

  /// Optional: max rows to compare for cuDF->Velox validation (0 = all).
  int64_t debugCudfToVeloxMaxRows{0};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation
  bool astExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};
};

} // namespace facebook::velox::cudf_velox
