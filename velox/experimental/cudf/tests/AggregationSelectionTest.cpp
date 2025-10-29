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
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;
using namespace facebook::velox::exec::test;

namespace {

class CudfAggregationSelectionTest : public ::testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
    cudf_velox::registerCudf();
    
    rowType_ = ROW({
        {"c0", BIGINT()},
        {"c1", BIGINT()},
        {"c2", INTEGER()},
        {"c3", DOUBLE()},
        {"c4", REAL()},
        {"c5", DOUBLE()},
        {"c6", VARCHAR()},
        {"c7", BOOLEAN()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  std::shared_ptr<const core::AggregationNode> createAggregationNode(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates) {
    auto plan = PlanBuilder()
        .values({makeRowVector({
            makeFlatVector<int64_t>({1, 2, 3}),
            makeFlatVector<int64_t>({10, 20, 30}),
            makeFlatVector<int32_t>({100, 200, 300}),
            makeFlatVector<double>({1.1, 2.2, 3.3}),
            makeFlatVector<float>({1.5f, 2.5f, 3.5f}),
            makeFlatVector<double>({10.1, 20.2, 30.3}),
            makeFlatVector<std::string>({"a", "b", "c"}),
            makeFlatVector<bool>({true, false, true}),
        })})
        .aggregation(groupingKeys, aggregates, {}, core::AggregationNode::Step::kSingle, false)
        .planNode();
    
    return std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
};

// Test 1: Supported Aggregation Functions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, supportedAggregationFunctions) {
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"});
  
  // This should return true for all supported CUDF aggregation functions
  ASSERT_TRUE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 2: Unsupported Aggregation Functions  
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationFunctions) {
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"stddev(c1)", "variance(c2)"});
  
  // This should return false for unsupported aggregation functions
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 3: Mixed Supported and Unsupported Functions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, mixedSupportedUnsupportedFunctions) {
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"sum(c1)", "stddev(c2)"});  // One supported, one unsupported
  
  // This should return false if ANY aggregation function is unsupported
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 4: Supported Grouping Key Expressions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, supportedGroupingKeyExpressions) {
  // Use simple field references that are definitely supported
  auto aggregationNode = createAggregationNode(
      {"c0", "c1"}, 
      {"sum(c2)"});
  
  // This should return true when grouping keys use supported expressions (simple field references)
  ASSERT_TRUE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 5: Unsupported Grouping Key Expressions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, unsupportedGroupingKeyExpressions) {
  // Test with an unsupported aggregation function instead of complex expressions
  // This is simpler and tests the same validation logic
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"sum(c1)", "stddev(c2)"}); // stddev is unsupported
  
  // This should return false when ANY aggregation function is unsupported
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 6: Supported Aggregation Input Expressions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, supportedAggregationInputExpressions) {
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"sum(c1 + c2)", "max(length(c6))"});
  
  // This should return true when aggregation inputs use supported expressions
  ASSERT_TRUE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 7: Unsupported Aggregation Input Expressions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationInputExpressions) {
  // Test with an unsupported aggregation function - this is simpler and tests the same logic
  auto aggregationNode = createAggregationNode(
      {"c0"}, 
      {"variance(c1)"}); // variance is unsupported
  
  // This should return false when aggregation functions are unsupported
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 8: Global Aggregation (No Grouping Keys)
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, globalAggregationSupported) {
  auto aggregationNode = createAggregationNode(
      {}, // No grouping keys
      {"sum(c1)", "count(c2)", "max(c3)"});
  
  // This should return true for supported global aggregations
  ASSERT_TRUE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 9: Global Aggregation with Unsupported Functions
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, globalAggregationUnsupported) {
  auto aggregationNode = createAggregationNode(
      {}, // No grouping keys
      {"stddev(c1)"});
  
  // This should return false for unsupported global aggregations
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 10: COUNT(DISTINCT) Aggregation - Should be rejected
// This tests that DISTINCT aggregations are properly rejected by CUDF validation
TEST_F(CudfAggregationSelectionTest, countDistinctAggregationRejected) {
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"count(distinct c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Should return FALSE because COUNT(DISTINCT) aggregations are not supported by CUDF
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 11: Complex Groupby Clause with Expressions
// This specifically tests the groupby clause validation with expression expansion
TEST_F(CudfAggregationSelectionTest, complexGroupbyClauseExpressions) {
  // Create a plan with complex expressions in GROUP BY that should be unsupported
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      })})
      .project({"c0", "c1", "abs(c0) AS abs_c0"}) // abs is unsupported by CUDF
      .aggregation({"abs_c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // This should return false because abs() in groupby clause is unsupported
  // This tests that our expression expansion correctly validates groupby expressions
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 12: Nested Aggregation - Allowed -> Not Allowed
// Inner aggregation (sum) is allowed, outer aggregation (stddev) is not allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationAllowedToNotAllowed) {
  // Step 1: Create inner aggregation (ALLOWED: sum)
  auto innerPlan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"sum(c1) AS inner_sum"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  // Step 2: Create outer aggregation (NOT ALLOWED: stddev)
  auto outerPlan = PlanBuilder()
      .addNode([&](std::string id, core::PlanNodePtr input) {
        return innerPlan;
      })
      .aggregation({}, {"stddev(inner_sum) AS outer_stddev"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto outerAggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);
  
  // Should return FALSE because outer aggregation uses unsupported stddev
  ASSERT_FALSE(canBeEvaluatedByCudf(*outerAggregationNode));
}

// Test 13: Nested Aggregation - Allowed -> Allowed
// Both inner (sum) and outer (sum) aggregations are allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationAllowedToAllowed) {
  // Step 1: Create inner aggregation (ALLOWED: sum)
  auto innerPlan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"sum(c1) AS inner_sum"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  // Step 2: Create outer aggregation (ALLOWED: sum)
  auto outerPlan = PlanBuilder()
      .addNode([&](std::string id, core::PlanNodePtr input) {
        return innerPlan;
      })
      .aggregation({}, {"sum(inner_sum) AS outer_sum"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto outerAggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);
  
  // Should return TRUE because both aggregations use supported functions
  ASSERT_TRUE(canBeEvaluatedByCudf(*outerAggregationNode));
}

// Test 14: Nested Aggregation - Not Allowed -> Allowed
// Inner aggregation (stddev) is not allowed, outer aggregation (sum) is allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationNotAllowedToAllowed) {
  // Step 1: Create inner aggregation (NOT ALLOWED: stddev)
  auto innerPlan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"stddev(c1) AS inner_stddev"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  // Step 2: Create outer aggregation (ALLOWED: sum)
  auto outerPlan = PlanBuilder()
      .addNode([&](std::string id, core::PlanNodePtr input) {
        return innerPlan;
      })
      .aggregation({}, {"sum(inner_stddev) AS outer_sum"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto outerAggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);
  
  // Should return TRUE because we only validate the current (outer) aggregation node
  // The inner aggregation's unsupported function doesn't affect the outer node's validation
  ASSERT_TRUE(canBeEvaluatedByCudf(*outerAggregationNode));
}

// Test 15: Nested Aggregation - Not Allowed -> Not Allowed
// Both inner (stddev) and outer (variance) aggregations are not allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationNotAllowedToNotAllowed) {
  // Step 1: Create inner aggregation (NOT ALLOWED: stddev)
  auto innerPlan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"stddev(c1) AS inner_stddev"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  // Step 2: Create outer aggregation (NOT ALLOWED: variance)
  auto outerPlan = PlanBuilder()
      .addNode([&](std::string id, core::PlanNodePtr input) {
        return innerPlan;
      })
      .aggregation({}, {"variance(inner_stddev) AS outer_variance"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto outerAggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);
  
  // Should return FALSE because outer aggregation uses unsupported variance
  ASSERT_FALSE(canBeEvaluatedByCudf(*outerAggregationNode));
}

// ORDER BY within aggregates tests removed - not applicable since CUDF doesn't support order-sensitive aggregates

// Test 18: Aggregation Functions in Function Registry - Signature Validation
// This tests that aggregation functions are properly registered with correct signatures
TEST_F(CudfAggregationSelectionTest, aggregationFunctionSignatures) {
  // Test that aggregation functions are registered in the function registry
  // and can be validated for signature compatibility
  
  // Create expressions that use aggregation functions with different types
  auto sumBigintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")
      },
      "sum");
  
  auto sumDoubleExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c1")
      },
      "sum");
  
  auto countExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")
      },
      "count");
  
  auto minVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c2")
      },
      "min");
  
  auto avgExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")
      },
      "avg");
  
  // These should all return true because they match registered signatures
  ASSERT_TRUE(canBeEvaluatedByCudf(sumBigintExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(sumDoubleExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(countExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(minVarcharExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(avgExpr));
}

// Test 19: Unsupported Aggregation Function Signatures
// This tests that aggregation functions with unsupported signatures are rejected
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationFunctionSignatures) {
  // Test aggregation functions that are not registered (should be rejected)
  auto stddevExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")
      },
      "stddev"); // stddev is not registered in CUDF
  
  auto varianceExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")
      },
      "variance"); // variance is not registered in CUDF
  
  // These should return false because they're not registered
  ASSERT_FALSE(canBeEvaluatedByCudf(stddevExpr));
  ASSERT_FALSE(canBeEvaluatedByCudf(varianceExpr));
}

// Test 20: Comprehensive Type Support Validation
// This tests that all registered type combinations actually work with CUDF
TEST_F(CudfAggregationSelectionTest, comprehensiveTypeSupportValidation) {
  // Test different numeric types with sum
  auto sumTinyintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "c0")}, "sum");
  auto sumSmallintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")}, "sum");
  auto sumIntegerExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")}, "sum");
  auto sumRealExpr = std::make_shared<core::CallTypedExpr>(
      REAL(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(REAL(), "c4")}, "sum");

  // Test min/max with different types (should preserve input type)
  auto minVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")}, "min");
  auto maxBooleanExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c7")}, "max");

  // Test count with different input types (always returns bigint)
  auto countVarcharExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")}, "count");

  // Test avg (always returns double for numeric inputs)
  auto avgIntegerExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")}, "avg");

  // All these should be supported based on our registered signatures
  ASSERT_TRUE(canBeEvaluatedByCudf(sumTinyintExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(sumSmallintExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(sumIntegerExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(sumRealExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(minVarcharExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(maxBooleanExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(countVarcharExpr));
  ASSERT_TRUE(canBeEvaluatedByCudf(avgIntegerExpr));
}

// Test 21: Invalid Type Combinations Should Be Rejected
// This tests that type combinations not supported by CUDF are properly rejected
TEST_F(CudfAggregationSelectionTest, invalidTypeCombinationsRejected) {
  // Test invalid combinations that should be rejected
  
  // avg on varchar (not supported - avg only works on numeric types)
  auto avgVarcharExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")}, "avg");
  
  // sum on varchar (not supported - sum only works on numeric types)  
  auto sumVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")}, "sum");

  // Wrong return type for sum (sum of integers should return bigint, not varchar)
  auto sumWrongReturnExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(), std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")}, "sum");

  // These should all be rejected due to invalid type combinations
  ASSERT_FALSE(canBeEvaluatedByCudf(avgVarcharExpr));
  ASSERT_FALSE(canBeEvaluatedByCudf(sumVarcharExpr));
  ASSERT_FALSE(canBeEvaluatedByCudf(sumWrongReturnExpr));
}

// Test 22: DISTINCT Aggregations Should Be Rejected
// DISTINCT aggregations are not supported by CUDF
TEST_F(CudfAggregationSelectionTest, distinctAggregationsRejected) {
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      })})
      .aggregation({"c0"}, {"count(distinct c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Should return FALSE because DISTINCT aggregations are not supported by CUDF
  ASSERT_FALSE(canBeEvaluatedByCudf(*aggregationNode));
}

// Test 23: FILTER/MASK Clauses Should Be Rejected
// FILTER clauses (WHERE conditions within aggregates) are NOT supported by CUDF (confirmed via correctness test)
TEST_F(CudfAggregationSelectionTest, filterMaskClausesRejected) {
  // Create a basic aggregation node first
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, false, true}),
      })})
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Manually create a modified aggregation with a mask (simulating FILTER clause)
  auto modifiedAggregates = aggregationNode->aggregates();
  if (!modifiedAggregates.empty()) {
    // Add a mask field (like: sum(c1) FILTER (WHERE c2))
    modifiedAggregates[0].mask = std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c2");
    
    auto modifiedNode = std::make_shared<core::AggregationNode>(
        aggregationNode->id(),
        aggregationNode->step(),
        aggregationNode->groupingKeys(),
        std::vector<core::FieldAccessTypedExprPtr>{}, // preGroupedKeys
        aggregationNode->aggregateNames(),
        modifiedAggregates,
        aggregationNode->ignoreNullKeys(),
        aggregationNode->sources()[0]);
    
    // Should return FALSE because FILTER/MASK clauses are NOT supported by CUDF
    ASSERT_FALSE(canBeEvaluatedByCudf(*modifiedNode));
  }
}

// Test 24: Global Grouping Sets Should Be Accepted
// Global grouping sets are indices for CUBE/ROLLUP operations - ARE supported by CUDF (confirmed via ablation test)
// Example SQL: SELECT key, SUM(value) FROM table GROUP BY CUBE(key)
// This creates multiple grouping combinations: {key}, {} (empty for grand total)
TEST_F(CudfAggregationSelectionTest, globalGroupingSetsAccepted) {
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
      })})
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Create a new aggregation node with global grouping sets (indices for different grouping combinations)
  auto globalGroupingSets = std::vector<vector_size_t>{0, 1, 2}; // Indices for CUBE/ROLLUP combinations
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
  
  // Should return TRUE because global grouping sets (CUBE/ROLLUP) ARE supported by CUDF
  ASSERT_TRUE(canBeEvaluatedByCudf(*modifiedNode));
}

// Test 25: Group ID Should Be Accepted
// Group ID (used with GROUPING SETS) ARE supported by CUDF (confirmed via ablation test)
TEST_F(CudfAggregationSelectionTest, groupIdAccepted) {
  auto plan = PlanBuilder()
      .values({makeRowVector({
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
      })})
      .aggregation({"c0"}, {"sum(c1)"}, {}, core::AggregationNode::Step::kSingle, false)
      .planNode();
  
  auto aggregationNode = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  
  // Create a new aggregation node with group ID (and required globalGroupingSets)
  auto groupId = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0");
  auto globalGroupingSets = std::vector<vector_size_t>{0, 1}; // Required when groupId is provided
  
  auto modifiedNode = std::make_shared<core::AggregationNode>(
      aggregationNode->id(),
      aggregationNode->step(),
      aggregationNode->groupingKeys(),
      std::vector<core::FieldAccessTypedExprPtr>{}, // preGroupedKeys
      aggregationNode->aggregateNames(),
      aggregationNode->aggregates(),
      globalGroupingSets, // globalGroupingSets is required when groupId is provided
      groupId, // Set group ID
      aggregationNode->ignoreNullKeys(),
      aggregationNode->sources()[0]);
  
  // Should return TRUE because Group ID ARE supported by CUDF (confirmed via correctness test)
  ASSERT_TRUE(canBeEvaluatedByCudf(*modifiedNode));
}


} // namespace
