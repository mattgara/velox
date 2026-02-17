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

#include "velox/experimental/cudf/CudfConfig.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, CudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfDebugOperatorFlow, "true"},
      {CudfConfig::kCudfDebugOperatorFlowSync, "true"},
      {CudfConfig::kCudfDebugOperatorFlowDeviceSyncPoint, "3"},
      {CudfConfig::kCudfDebugHashAggFakeGroupbyMode, "2"},
      {CudfConfig::kCudfDebugHashAggProbeDumpDir, "/tmp/hashagg_dumps"},
      {CudfConfig::kCudfDebugHashAggProbeDumpMaxRows, "1234"},
      {CudfConfig::kCudfDebugHashAggDumpDir, "/tmp/hashagg_full_dumps"},
      {CudfConfig::kCudfDebugHashAggDumpMaxRows, "4321"},
      {CudfConfig::kCudfDebugHashAggDecimalCpuAggregateMode, "1"},
      {CudfConfig::kCudfDebugCudfToVeloxValidate, "true"},
      {CudfConfig::kCudfDebugCudfToVeloxMaxRows, "2048"},
      {CudfConfig::kCudfDebugSerdeValidate, "true"},
      {CudfConfig::kCudfDebugSerdeValidateMaxRows, "4096"},
      {CudfConfig::kCudfDebugHashAggEndToEndValidate, "true"},
      {CudfConfig::kCudfDebugHashAggEndToEndMaxRows, "123456"},
      {CudfConfig::kCudfDebugHashAggEndToEndBatchRows, "6543"},
      {CudfConfig::kCudfDebugHashAggExpectedPath, "/tmp/hashagg_expected.csv"},
      {CudfConfig::kCudfDebugHashAggStateRoundtripValidate, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.debugOperatorFlow, true);
  ASSERT_EQ(config.debugOperatorFlowSync, true);
  ASSERT_EQ(config.debugOperatorFlowDeviceSyncPoint, 3);
  ASSERT_EQ(config.debugHashAggFakeGroupbyMode, 2);
  ASSERT_EQ(config.debugHashAggProbeDumpDir, "/tmp/hashagg_dumps");
  ASSERT_EQ(config.debugHashAggProbeDumpMaxRows, 1234);
  ASSERT_EQ(config.debugHashAggDumpDir, "/tmp/hashagg_full_dumps");
  ASSERT_EQ(config.debugHashAggDumpMaxRows, 4321);
  ASSERT_EQ(config.debugHashAggDecimalCpuAggregateMode, 1);
  ASSERT_EQ(config.debugCudfToVeloxValidate, true);
  ASSERT_EQ(config.debugCudfToVeloxMaxRows, 2048);
  ASSERT_EQ(config.debugSerdeValidate, true);
  ASSERT_EQ(config.debugSerdeValidateMaxRows, 4096);
  ASSERT_EQ(config.debugHashAggEndToEndValidate, true);
  ASSERT_EQ(config.debugHashAggEndToEndMaxRows, 123456);
  ASSERT_EQ(config.debugHashAggEndToEndBatchRows, 6543);
  ASSERT_EQ(config.debugHashAggExpectedPath, "/tmp/hashagg_expected.csv");
  ASSERT_EQ(config.debugHashAggStateRoundtripValidate, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.allowCpuFallback, false);
}

TEST(ConfigTest, CudfConfigAllDeviceSyncProbes) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfDebugOperatorFlowDeviceSyncPoint, "-1"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.debugOperatorFlowDeviceSyncPoint, -1);
}
} // namespace facebook::velox::cudf_velox::test
