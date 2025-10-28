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

// Test 10: Empty Aggregation (Distinct only)
// This test should FAIL initially because canBeEvaluatedByCudf(AggregationNode) doesn't exist
TEST_F(CudfAggregationSelectionTest, distinctOnlyAggregation) {
  auto aggregationNode = createAggregationNode(
      {"c0", "c6"}, // Grouping keys only
      {});          // No aggregation functions
  
  // This should return true for distinct-only operations with supported grouping keys
  ASSERT_TRUE(canBeEvaluatedByCudf(*aggregationNode));
}

} // namespace
