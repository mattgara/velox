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

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class ToCudfSelectionTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  bool wasCudfHashAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    // Check if any CUDF aggregation operators were used
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfAggregation") {
          return true;
        }
      }
    }
    return false;
  }

  bool wasRegularHashAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    // Check if any regular aggregation operators were used
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "Aggregation") {
          return true;
        }
      }
    }
    return false;
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(),
           DOUBLE(),
           VARCHAR()})};
};

// Test 1: Supported Aggregation Should Use CudfHashAggregation
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, supportedAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"}, 
                      {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), count(c2), min(c3), max(c4), avg(c5) FROM tmp GROUP BY c0");

  // Should use CudfHashAggregation for supported aggregations
  ASSERT_TRUE(wasCudfHashAggregationUsed(task));
  ASSERT_FALSE(wasRegularHashAggregationUsed(task));
}

// Test 2: Unsupported Aggregation Should Fall Back to Regular HashAggregation  
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, unsupportedAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Note: stddev and variance are not supported by CudfHashAggregation
  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"stddev(c1)", "variance(c2)"}, {}, core::AggregationNode::Step::kSingle, false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  // TODO: Add CUDF config once available
                  .plan(plan)
                  .assertResults("SELECT c0, stddev(c1), variance(c2) FROM tmp GROUP BY c0");

  // Should fall back to regular HashAggregation for unsupported aggregations
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 3: Mixed Supported/Unsupported Should Fall Back
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, mixedSupportFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Mix of supported (sum) and unsupported (stddev) functions
  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"sum(c1)", "stddev(c2)"}, {}, core::AggregationNode::Step::kSingle, false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  // TODO: Add CUDF config once available
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  // Should fall back to regular HashAggregation when ANY function is unsupported
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 4: Supported Global Aggregation Should Use CudfHashAggregation
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, supportedGlobalAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({}, {"sum(c1)", "count(c2)", "max(c3)"}, {}, core::AggregationNode::Step::kSingle, false) // Global aggregation
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(c1), count(c2), max(c3) FROM tmp");

  // Should use CudfHashAggregation for supported global aggregations
  ASSERT_TRUE(wasCudfHashAggregationUsed(task));
  ASSERT_FALSE(wasRegularHashAggregationUsed(task));
}

// Test 5: Unsupported Global Aggregation Should Fall Back
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, unsupportedGlobalAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({}, {"stddev(c1)"}, {}, core::AggregationNode::Step::kSingle, false) // Global aggregation with unsupported function
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  // TODO: Add CUDF config once available
                  .plan(plan)
                  .assertResults("SELECT stddev(c1) FROM tmp");

  // Should fall back to regular HashAggregation for unsupported global aggregations
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 6: Supported Grouping Key Expressions Should Use CudfHashAggregation
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, supportedGroupingKeyExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c0 + c1 as key1", "length(c6) as key2"})
                  .aggregation({"key1", "key2"}, {"sum(c2)"}, {}, core::AggregationNode::Step::kSingle, false) // Use projected expressions
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT c0 + c1, length(c6), sum(c2) FROM tmp GROUP BY c0 + c1, length(c6)");

  // Should use CudfHashAggregation when grouping key expressions are supported
  ASSERT_TRUE(wasCudfHashAggregationUsed(task));
  ASSERT_FALSE(wasRegularHashAggregationUsed(task));
}

// Test 7: Unsupported Aggregation Functions Should Fall Back
// This test verifies that unsupported aggregation functions cause fallback to regular aggregation
TEST_F(ToCudfSelectionTest, unsupportedAggregationFunctionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Create a plan with an unsupported aggregation function that should cause fallback
  // This tests the expression expansion validation
  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"sum(c1)", "stddev(c2)"}, {}, core::AggregationNode::Step::kSingle, false) // stddev is unsupported
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  // Should fall back to regular HashAggregation when unsupported aggregation functions are used
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 7b: Complex Grouping Key Expressions Should Fall Back (Expression Expansion Test)
// This test verifies that complex expressions in grouping keys are properly detected via expression expansion
TEST_F(ToCudfSelectionTest, complexGroupingKeyExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Create a plan with complex grouping key expressions that should be unsupported
  // This tests the expression expansion functionality  
  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "abs(c0) AS complex_key"}) // abs function (likely unsupported)
                  .aggregation({"complex_key"}, {"sum(c2)"}, {}, core::AggregationNode::Step::kSingle, false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT abs(c0), sum(c2) FROM tmp GROUP BY abs(c0)");

  // Should fall back to regular HashAggregation when complex expressions are used in grouping keys
  // This tests that our expression expansion correctly identifies the underlying abs(c0) expression
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 8: Supported Aggregation Input Expressions Should Use CudfHashAggregation
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, supportedAggregationInputExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"sum(c1)", "max(c2)"}, {}, core::AggregationNode::Step::kSingle, false) // Simple field references in aggregation inputs
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), max(c2) FROM tmp GROUP BY c0");

  // Should use CudfHashAggregation when aggregation input expressions are supported
  ASSERT_TRUE(wasCudfHashAggregationUsed(task));
  ASSERT_FALSE(wasRegularHashAggregationUsed(task));
}

// Test 9: Unsupported Aggregation Input Expressions Should Fall Back
// This test should FAIL initially because currently ALL aggregations go to CUDF unconditionally
TEST_F(ToCudfSelectionTest, unsupportedAggregationInputExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Create a plan with an unsupported aggregation function
  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"sum(c1)", "variance(c2)"}, {}, core::AggregationNode::Step::kSingle, false) // variance is unsupported
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  // Should fall back to regular HashAggregation when aggregation functions are unsupported
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

// Test 10: CUDF Disabled Should Always Use Regular HashAggregation
// This test should PASS initially as it tests the fallback when CUDF is disabled
TEST_F(ToCudfSelectionTest, cudfDisabledUsesRegularAggregation) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation({"c0"}, {"sum(c1)", "count(c2)"}, {}, core::AggregationNode::Step::kSingle, false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                  .config("cudf.enabled", false)
                  .plan(plan)
                  .assertResults("SELECT c0, sum(c1), count(c2) FROM tmp GROUP BY c0");

  // Should use regular HashAggregation when CUDF is disabled
  ASSERT_FALSE(wasCudfHashAggregationUsed(task));
  ASSERT_TRUE(wasRegularHashAggregationUsed(task));
}

} // namespace facebook::velox::exec::test
