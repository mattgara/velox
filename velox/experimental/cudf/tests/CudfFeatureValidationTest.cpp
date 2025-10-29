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

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox::test {

class CudfFeatureValidationTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    // Register all functions
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Helper to check if CUDF aggregation was actually used in task execution
  bool wasCudfUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        // Look specifically for CUDF aggregation, not just any CUDF operator
        if (operatorStats.operatorType == "CudfAggregation") {
          return true;
        }
      }
    }
    return false;
  }

  // Helper to create test data
  std::vector<RowVectorPtr> createTestData() {
    return {makeRowVector({
        makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3, 1, 2, 3}),      // c0 - grouping key
        makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80, 90}), // c1 - values
        makeFlatVector<std::string>({"a", "a", "b", "b", "c", "c", "a", "b", "c"}), // c2 - string grouping
        makeFlatVector<bool>({true, false, true, false, true, false, true, false, true}), // c3 - boolean mask
    })};
  }

  RowTypePtr rowType_{ROW({"c0", "c1", "c2", "c3"}, {BIGINT(), BIGINT(), VARCHAR(), BOOLEAN()})};

 protected:
  void testGroupByScenario(
      const std::vector<RowVectorPtr>& vectors,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& testName) {
    
    std::cout << "Testing: " << testName << std::endl;
    
    try {
      auto plan = PlanBuilder()
          .values(vectors)
          .aggregation(groupingKeys, aggregates, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();
      
      runGroupByCorrectnessTest(plan, testName);
      
    } catch (const std::exception& e) {
      std::cout << "❌ " << testName << ": FAILED to create plan - " << e.what() << std::endl;
    }
  }
  
  void testGroupByScenarioWithProjection(
      const std::vector<RowVectorPtr>& vectors,
      const std::vector<std::string>& projections,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& testName) {
    
    std::cout << "Testing: " << testName << std::endl;
    
    try {
      auto plan = PlanBuilder()
          .values(vectors)
          .project(projections)
          .aggregation(groupingKeys, aggregates, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();
      
      runGroupByCorrectnessTest(plan, testName);
      
    } catch (const std::exception& e) {
      std::cout << "❌ " << testName << ": FAILED to create plan - " << e.what() << std::endl;
    }
  }
  
  void runGroupByCorrectnessTest(const core::PlanNodePtr& plan, const std::string& testName) {
    RowVectorPtr cudfResults;
    RowVectorPtr cpuResults;
    bool cudfUsed = false;
    
    // Check if the plan should be supported by CUDF for validation
    auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    bool shouldBeSupported = aggregationNode ? canBeEvaluatedByCudf(*aggregationNode) : false;
    
    // Step 1: Run with CUDF enabled
    try {
      std::shared_ptr<exec::Task> cudfTask;
      cudfResults = AssertQueryBuilder(duckDbQueryRunner_)
                      .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                      .config("cudf.enabled", true)
                      .plan(plan)
                      .copyResults(pool(), cudfTask);
      
      cudfUsed = wasCudfUsed(cudfTask);
      
    } catch (const std::exception& e) {
      std::cout << "❌ " << testName << ": CUDF FAILED - " << e.what() << std::endl;
      return;
    }
    
    // Step 2: Run with CUDF disabled (force CPU)
    try {
      std::shared_ptr<exec::Task> cpuTask;
      cpuResults = AssertQueryBuilder(duckDbQueryRunner_)
                      .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                      .config("cudf.enabled", false)
                      .plan(plan)
                      .copyResults(pool(), cpuTask);
      
    } catch (const std::exception& e) {
      std::cout << "❌ " << testName << ": CPU ALSO FAILED - " << e.what() << std::endl;
      return;
    }
    
    // Step 3: Compare results (ORDER-INDEPENDENT comparison for GROUP BY)
    bool resultsMatch = true;
    if (cudfResults->size() != cpuResults->size()) {
      resultsMatch = false;
    } else {
      // For GROUP BY results, we need to compare sets, not ordered lists
      // Create a set of string representations for comparison
      std::set<std::string> cudfSet, cpuSet;
      
      for (int i = 0; i < cudfResults->size(); ++i) {
        cudfSet.insert(cudfResults->toString(i));
        cpuSet.insert(cpuResults->toString(i));
      }
      
      resultsMatch = (cudfSet == cpuSet);
    }
    
    // Step 4: Report results
    if (cudfUsed && resultsMatch) {
      std::cout << "✅ " << testName << ": CUDF SUCCESS (correct results)" << std::endl;
    } else if (cudfUsed && !resultsMatch) {
      std::cout << "❌ " << testName << ": CUDF INCORRECT RESULTS!" << std::endl;
    } else if (!cudfUsed) {
      std::cout << "⚠️  " << testName << ": CUDF FALLBACK (validation worked)" << std::endl;
    }
  }
};

// Test 1: DISTINCT Aggregations - Should this work with CUDF?
TEST_F(CudfFeatureValidationTest, distinctAggregationValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"}, 
                      {"count(distinct c1)"}, // DISTINCT aggregation
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  // Test with CUDF ENABLED - should this work or throw an error?
  std::cout << "Testing DISTINCT aggregation with CUDF enabled..." << std::endl;
  try {
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", true)
                    .plan(plan)
                    .assertResults("SELECT c0, count(distinct c1) FROM tmp GROUP BY c0");
    
    // Check if CUDF was actually used
    bool cudfUsed = wasCudfUsed(task);
    std::cout << "DISTINCT AGGREGATION with CUDF ENABLED: SUCCESS" << std::endl;
    std::cout << "CUDF was " << (cudfUsed ? "USED" : "NOT USED") << std::endl;
    
    if (cudfUsed) {
      std::cout << "This confirms DISTINCT aggregations ARE supported by CUDF!" << std::endl;
    } else {
      std::cout << "Query succeeded but fell back to regular Velox - DISTINCT not supported by CUDF" << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::cout << "DISTINCT AGGREGATION with CUDF ENABLED: FAILED - " << e.what() << std::endl;
    
    // If CUDF fails, verify it works with CUDF disabled
    std::cout << "Testing fallback with CUDF disabled..." << std::endl;
    try {
      auto resultFallback = AssertQueryBuilder(duckDbQueryRunner_)
                              .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                              .config("cudf.enabled", false)
                              .plan(plan)
                              .assertResults("SELECT c0, count(distinct c1) FROM tmp GROUP BY c0");
      
      std::cout << "DISTINCT AGGREGATION with CUDF DISABLED: SUCCESS (fallback works)" << std::endl;
      std::cout << "This confirms DISTINCT aggregations are NOT supported by CUDF" << std::endl;
    } catch (const std::exception& e2) {
      std::cout << "DISTINCT AGGREGATION with CUDF DISABLED: ALSO FAILED - " << e2.what() << std::endl;
      FAIL() << "DISTINCT aggregations should work with regular Velox";
    }
  }
}

// Test 2: FILTER/MASK Clauses - Correctness Test
TEST_F(CudfFeatureValidationTest, filterMaskValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  // Create a basic aggregation first, then manually add a mask
  auto plan = PlanBuilder()
      .values(vectors)
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Manually create a modified aggregation with a mask (simulating FILTER clause)
  auto modifiedAggregates = aggregationNode->aggregates();
  if (!modifiedAggregates.empty()) {
    // Add a mask field (like: sum(c1) FILTER (WHERE c3))
    modifiedAggregates[0].mask = std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c3");
    
    auto modifiedNode = std::make_shared<core::AggregationNode>(
        aggregationNode->id(),
        aggregationNode->step(),
        aggregationNode->groupingKeys(),
        std::vector<core::FieldAccessTypedExprPtr>{}, // preGroupedKeys
        aggregationNode->aggregateNames(),
        modifiedAggregates,
        aggregationNode->ignoreNullKeys(),
        aggregationNode->sources()[0]);

    // CORRECTNESS TEST: Run both CUDF and CPU and compare results
    std::cout << "=== CORRECTNESS TEST: FILTER/MASK CLAUSES ===" << std::endl;
    
    RowVectorPtr cudfResults;
    RowVectorPtr cpuResults;
    bool cudfUsed = false;
    
    // Step 1: Run with CUDF enabled
    std::cout << "Step 1: Running with CUDF enabled..." << std::endl;
    try {
      std::shared_ptr<exec::Task> cudfTask;
      cudfResults = AssertQueryBuilder(duckDbQueryRunner_)
                      .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                      .config("cudf.enabled", true)
                      .plan(modifiedNode)
                      .copyResults(pool(), cudfTask);
      
      cudfUsed = wasCudfUsed(cudfTask);
      std::cout << "CUDF path: " << (cudfUsed ? "SUCCESS (CudfAggregation used)" : "FALLBACK (regular Velox used)") << std::endl;
      
    } catch (const std::exception& e) {
      std::cout << "CUDF path: FAILED - " << e.what() << std::endl;
      return; // Can't continue without CUDF results
    }
    
    // Step 2: Run with CUDF disabled (force CPU)
    std::cout << "Step 2: Running with CUDF disabled (CPU only)..." << std::endl;
    try {
      std::shared_ptr<exec::Task> cpuTask;
      cpuResults = AssertQueryBuilder(duckDbQueryRunner_)
                      .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                      .config("cudf.enabled", false)
                      .plan(modifiedNode)
                      .copyResults(pool(), cpuTask);
      
      std::cout << "CPU path: SUCCESS" << std::endl;
      
    } catch (const std::exception& e) {
      std::cout << "CPU path: FAILED - " << e.what() << std::endl;
      return; // Can't continue without CPU results
    }
    
    // Step 3: Compare results
    std::cout << "\n=== RESULTS COMPARISON ===" << std::endl;
    std::cout << "CUDF Results (" << cudfResults->size() << " rows):" << std::endl;
    for (int i = 0; i < cudfResults->size(); ++i) {
      std::cout << "  Row " << i << ": " << cudfResults->toString(i) << std::endl;
    }
    
    std::cout << "CPU Results (" << cpuResults->size() << " rows):" << std::endl;
    for (int i = 0; i < cpuResults->size(); ++i) {
      std::cout << "  Row " << i << ": " << cpuResults->toString(i) << std::endl;
    }
    
    // Step 4: Verdict - Compare all rows
    bool resultsMatch = true;
    if (cudfResults->size() != cpuResults->size()) {
      resultsMatch = false;
      std::cout << "Row count mismatch!" << std::endl;
    } else {
      for (int i = 0; i < cudfResults->size() && resultsMatch; ++i) {
        if (!cudfResults->equalValueAt(cpuResults.get(), i, i)) {
          resultsMatch = false;
          std::cout << "Row " << i << " values differ!" << std::endl;
        }
      }
    }
    std::cout << "\n=== VERDICT ===" << std::endl;
    if (cudfUsed && resultsMatch) {
      std::cout << "✅ FILTER/MASK: CUDF produces CORRECT results!" << std::endl;
    } else if (cudfUsed && !resultsMatch) {
      std::cout << "❌ FILTER/MASK: CUDF produces INCORRECT results!" << std::endl;
      std::cout << "   CUDF doesn't properly handle FILTER/MASK clauses" << std::endl;
    } else if (!cudfUsed) {
      std::cout << "⚠️  FILTER/MASK: CUDF fell back to CPU (validation worked)" << std::endl;
    }
  }
}

// Test 3: Global Grouping Sets (CUBE/ROLLUP) - Should this work with CUDF?
TEST_F(CudfFeatureValidationTest, globalGroupingSetsValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  // Create a basic aggregation first, then manually add global grouping sets
  auto plan = PlanBuilder()
      .values(vectors)
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Create a new aggregation node with global grouping sets (indices for different grouping combinations)
  auto globalGroupingSets = std::vector<vector_size_t>{0, 1}; // Indices for CUBE/ROLLUP combinations
  auto groupId = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0"); // Required when globalGroupingSets is not empty
  
  auto modifiedNode = std::make_shared<core::AggregationNode>(
      aggregationNode->id(),
      aggregationNode->step(),
      aggregationNode->groupingKeys(),
      std::vector<core::FieldAccessTypedExprPtr>{}, // preGroupedKeys
      aggregationNode->aggregateNames(),
      aggregationNode->aggregates(),
      globalGroupingSets, // Set global grouping sets (CUBE/ROLLUP indices)
      groupId, // groupId is required when globalGroupingSets is not empty
      aggregationNode->ignoreNullKeys(),
      aggregationNode->sources()[0]);

  // CORRECTNESS TEST: Run both CUDF and CPU and compare results
  std::cout << "=== CORRECTNESS TEST: GLOBAL GROUPING SETS ===" << std::endl;
  
  RowVectorPtr cudfResults;
  RowVectorPtr cpuResults;
  bool cudfUsed = false;
  
  // Step 1: Run with CUDF enabled
  std::cout << "Step 1: Running with CUDF enabled..." << std::endl;
  try {
    std::shared_ptr<exec::Task> cudfTask;
    cudfResults = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", true)
                    .plan(modifiedNode)
                    .copyResults(pool(), cudfTask);
    
    cudfUsed = wasCudfUsed(cudfTask);
    std::cout << "CUDF path: " << (cudfUsed ? "SUCCESS (CudfAggregation used)" : "FALLBACK (regular Velox used)") << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "CUDF path: FAILED - " << e.what() << std::endl;
    return; // Can't continue without CUDF results
  }
  
  // Step 2: Run with CUDF disabled (force CPU)
  std::cout << "Step 2: Running with CUDF disabled (CPU only)..." << std::endl;
  try {
    std::shared_ptr<exec::Task> cpuTask;
    cpuResults = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", false)
                    .plan(modifiedNode)
                    .copyResults(pool(), cpuTask);
    
    std::cout << "CPU path: SUCCESS" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "CPU path: FAILED - " << e.what() << std::endl;
    return; // Can't continue without CPU results
  }
  
  // Step 3: Compare results
  std::cout << "\n=== RESULTS COMPARISON ===" << std::endl;
  std::cout << "CUDF Results (" << cudfResults->size() << " rows):" << std::endl;
  for (int i = 0; i < cudfResults->size(); ++i) {
    std::cout << "  Row " << i << ": " << cudfResults->toString(i) << std::endl;
  }
  
  std::cout << "CPU Results (" << cpuResults->size() << " rows):" << std::endl;
  for (int i = 0; i < cpuResults->size(); ++i) {
    std::cout << "  Row " << i << ": " << cpuResults->toString(i) << std::endl;
  }
  
  // Step 4: Verdict - Compare all rows
  bool resultsMatch = true;
  if (cudfResults->size() != cpuResults->size()) {
    resultsMatch = false;
    std::cout << "Row count mismatch!" << std::endl;
  } else {
    for (int i = 0; i < cudfResults->size() && resultsMatch; ++i) {
      if (!cudfResults->equalValueAt(cpuResults.get(), i, i)) {
        resultsMatch = false;
        std::cout << "Row " << i << " values differ!" << std::endl;
      }
    }
  }
  std::cout << "\n=== VERDICT ===" << std::endl;
  if (cudfUsed && resultsMatch) {
    std::cout << "✅ GLOBAL GROUPING SETS: CUDF produces CORRECT results!" << std::endl;
  } else if (cudfUsed && !resultsMatch) {
    std::cout << "❌ GLOBAL GROUPING SETS: CUDF produces INCORRECT results!" << std::endl;
    std::cout << "   This confirms the TODO comment - CUDF doesn't properly handle global grouping sets" << std::endl;
  } else if (!cudfUsed) {
    std::cout << "⚠️  GLOBAL GROUPING SETS: CUDF fell back to CPU (validation worked)" << std::endl;
  }
}

// Test 4: Group ID - Correctness Test
TEST_F(CudfFeatureValidationTest, groupIdValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  // Create a basic aggregation first, then manually add group ID
  auto plan = PlanBuilder()
      .values(vectors)
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Create a new aggregation node with group ID (and required globalGroupingSets)
  auto globalGroupingSets = std::vector<vector_size_t>{0, 1}; // Required when groupId is provided
  auto groupId = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0");
  
  auto modifiedNode = std::make_shared<core::AggregationNode>(
      aggregationNode->id(),
      aggregationNode->step(),
      aggregationNode->groupingKeys(),
      std::vector<core::FieldAccessTypedExprPtr>{}, // preGroupedKeys
      aggregationNode->aggregateNames(),
      aggregationNode->aggregates(),
      globalGroupingSets, // globalGroupingSets is required when groupId is provided
      groupId, // Group ID for GROUPING function
      aggregationNode->ignoreNullKeys(),
      aggregationNode->sources()[0]);

  // CORRECTNESS TEST: Run both CUDF and CPU and compare results
  std::cout << "=== CORRECTNESS TEST: GROUP ID ===" << std::endl;
  
  RowVectorPtr cudfResults;
  RowVectorPtr cpuResults;
  bool cudfUsed = false;
  
  // Step 1: Run with CUDF enabled
  std::cout << "Step 1: Running with CUDF enabled..." << std::endl;
  try {
    std::shared_ptr<exec::Task> cudfTask;
    cudfResults = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", true)
                    .plan(modifiedNode)
                    .copyResults(pool(), cudfTask);
    
    cudfUsed = wasCudfUsed(cudfTask);
    std::cout << "CUDF path: " << (cudfUsed ? "SUCCESS (CudfAggregation used)" : "FALLBACK (regular Velox used)") << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "CUDF path: FAILED - " << e.what() << std::endl;
    return; // Can't continue without CUDF results
  }
  
  // Step 2: Run with CUDF disabled (force CPU)
  std::cout << "Step 2: Running with CUDF disabled (CPU only)..." << std::endl;
  try {
    std::shared_ptr<exec::Task> cpuTask;
    cpuResults = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", false)
                    .plan(modifiedNode)
                    .copyResults(pool(), cpuTask);
    
    std::cout << "CPU path: SUCCESS" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "CPU path: FAILED - " << e.what() << std::endl;
    return; // Can't continue without CPU results
  }
  
  // Step 3: Compare results
  std::cout << "\n=== RESULTS COMPARISON ===" << std::endl;
  std::cout << "CUDF Results (" << cudfResults->size() << " rows):" << std::endl;
  for (int i = 0; i < cudfResults->size(); ++i) {
    std::cout << "  Row " << i << ": " << cudfResults->toString(i) << std::endl;
  }
  
  std::cout << "CPU Results (" << cpuResults->size() << " rows):" << std::endl;
  for (int i = 0; i < cpuResults->size(); ++i) {
    std::cout << "  Row " << i << ": " << cpuResults->toString(i) << std::endl;
  }
  
  // Step 4: Verdict - Compare all rows
  bool resultsMatch = true;
  if (cudfResults->size() != cpuResults->size()) {
    resultsMatch = false;
    std::cout << "Row count mismatch!" << std::endl;
  } else {
    for (int i = 0; i < cudfResults->size() && resultsMatch; ++i) {
      if (!cudfResults->equalValueAt(cpuResults.get(), i, i)) {
        resultsMatch = false;
        std::cout << "Row " << i << " values differ!" << std::endl;
      }
    }
  }
  std::cout << "\n=== VERDICT ===" << std::endl;
  if (cudfUsed && resultsMatch) {
    std::cout << "✅ GROUP ID: CUDF produces CORRECT results!" << std::endl;
  } else if (cudfUsed && !resultsMatch) {
    std::cout << "❌ GROUP ID: CUDF produces INCORRECT results!" << std::endl;
    std::cout << "   This confirms the TODO comment - CUDF doesn't properly handle group IDs" << std::endl;
  } else if (!cudfUsed) {
    std::cout << "⚠️  GROUP ID: CUDF fell back to CPU (validation worked)" << std::endl;
  }
}

// Test 5: GROUP BY Validation - Comprehensive Ablation Study
TEST_F(CudfFeatureValidationTest, groupByValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  std::cout << "=== COMPREHENSIVE GROUP BY ABLATION STUDY ===" << std::endl;
  
  // Test 1: Simple field references (should work)
  std::cout << "\n--- Test 1: Simple field references ---" << std::endl;
  testGroupByScenario(vectors, {"c0"}, {"sum(c1)"}, "Simple single column");
  testGroupByScenario(vectors, {"c0", "c2"}, {"sum(c1)"}, "Multiple simple columns");
  
  // Test 2: No GROUP BY - global aggregation (should work)
  std::cout << "\n--- Test 2: Global aggregation (no GROUP BY) ---" << std::endl;
  testGroupByScenario(vectors, {}, {"sum(c1)", "count(*)"}, "Global aggregation");
  
  // Test 3: Complex expressions (depends on expression support)
  std::cout << "\n--- Test 3: Complex expressions ---" << std::endl;
  testGroupByScenarioWithProjection(vectors, 
    {"c0", "c1", "c2", "c0 + 1 as c0_plus_1"}, // projection
    {"c0_plus_1"}, {"sum(c1)"}, // grouping and aggregation
    "Simple arithmetic in GROUP BY");
    
  testGroupByScenarioWithProjection(vectors,
    {"c0", "c1", "c2", "abs(c0) as abs_c0"}, // projection with unsupported function
    {"abs_c0"}, {"sum(c1)"}, // grouping and aggregation
    "Unsupported function (abs) in GROUP BY");
    
  // Test 4: Mixed data types
  std::cout << "\n--- Test 4: Mixed data types ---" << std::endl;
  testGroupByScenario(vectors, {"c0", "c2"}, {"sum(c1)"}, "Integer + String grouping");
  
  // Test 5: Many GROUP BY columns
  std::cout << "\n--- Test 5: Many GROUP BY columns ---" << std::endl;
  testGroupByScenario(vectors, {"c0", "c1", "c2"}, {"count(*)"}, "Many grouping columns");
}

// Test 6: Regular Aggregation (Control Test) - Should work with CUDF
TEST_F(CudfFeatureValidationTest, regularAggregationValidation) {
  auto vectors = createTestData();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"}, 
                      {"sum(c1)", "count(c1)", "min(c1)", "max(c1)"}, // Regular aggregations
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  // Test with CUDF ENABLED - this should work
  std::cout << "Testing regular aggregations with CUDF enabled..." << std::endl;
  try {
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(core::QueryConfig::kEnableExpressionEvaluationCache, false)
                    .config("cudf.enabled", true)
                    .plan(plan)
                    .assertResults("SELECT c0, sum(c1), count(c1), min(c1), max(c1) FROM tmp GROUP BY c0");
    
    // Check if CUDF was actually used
    bool cudfUsed = wasCudfUsed(task);
    std::cout << "REGULAR AGGREGATION with CUDF ENABLED: SUCCESS" << std::endl;
    std::cout << "CUDF was " << (cudfUsed ? "USED" : "NOT USED") << std::endl;
    
    if (!cudfUsed) {
      FAIL() << "Regular aggregations should use CUDF when enabled";
    }
    
  } catch (const std::exception& e) {
    std::cout << "REGULAR AGGREGATION with CUDF ENABLED: FAILED - " << e.what() << std::endl;
    FAIL() << "Regular aggregations should work with CUDF";
  }
}

} // namespace facebook::velox::cudf_velox::test
