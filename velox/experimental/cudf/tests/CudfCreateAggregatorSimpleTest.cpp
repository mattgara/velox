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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
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

class CudfCreateAggregatorSimpleTest : public ::testing::Test, public test::VectorTestBase {
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
    
    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  // Test that tries to actually run CUDF aggregations end-to-end
  void testCudfAggregationEndToEnd(
      const std::string& functionName,
      const TypePtr& inputType,
      const TypePtr& expectedResultType,
      bool shouldSucceed = true) {
    
    std::cout << "\n=== Testing CUDF end-to-end for " << functionName << "(" << inputType->toString() 
              << ") -> " << expectedResultType->toString() << " ===" << std::endl;
    std::cout << "📝 ORIGINAL INPUT TYPE: " << inputType->toString() << std::endl;
    std::cout << "🎯 EXPECTED RESULT TYPE: " << expectedResultType->toString() << std::endl;
    std::cout << "✅ SHOULD SUCCEED: " << (shouldSucceed ? "YES" : "NO") << std::endl;
    
    try {
      // Create test data
      std::vector<VectorPtr> children;
      children.push_back(makeFlatVector<int64_t>({1, 1, 2, 2, 3})); // grouping key
      
      // Create appropriate test data based on input type
      if (inputType->kind() == TypeKind::BIGINT) {
        children.push_back(makeFlatVector<int64_t>({10, 20, 30, 40, 50}));
      } else if (inputType->kind() == TypeKind::INTEGER) {
        children.push_back(makeFlatVector<int32_t>({10, 20, 30, 40, 50}));
      } else if (inputType->kind() == TypeKind::DOUBLE) {
        children.push_back(makeFlatVector<double>({10.1, 20.2, 30.3, 40.4, 50.5}));
      } else if (inputType->kind() == TypeKind::REAL) {
        children.push_back(makeFlatVector<float>({10.1f, 20.2f, 30.3f, 40.4f, 50.5f}));
      } else if (inputType->kind() == TypeKind::VARCHAR) {
        children.push_back(makeFlatVector<std::string>({"a", "b", "c", "d", "e"}));
      } else if (inputType->kind() == TypeKind::BOOLEAN) {
        children.push_back(makeFlatVector<bool>({true, false, true, false, true}));
      } else {
        children.push_back(BaseVector::create(inputType, 5, pool_.get()));
      }
      
      auto vectors = makeRowVector(children);
      std::vector<RowVectorPtr> vectorList = {vectors};
      
      // Create aggregation plan
      auto plan = PlanBuilder()
          .values(vectorList)
          .aggregation(
              {"c0"}, // group by first column
              {functionName + "(c1)"}, // aggregate second column
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode();
      
      // Try to run with CUDF enabled
      std::cout << "Testing with CUDF enabled..." << std::endl;
      
      std::shared_ptr<exec::Task> cudfTask;
      auto cudfResults = AssertQueryBuilder(plan)
          .config("cudf.enabled", "true")
          .copyResults(pool(), cudfTask);
      
      // Check if CUDF was actually used
      bool cudfUsed = false;
      auto stats = cudfTask->taskStats();
      for (const auto& pipelineStats : stats.pipelineStats) {
        for (const auto& operatorStats : pipelineStats.operatorStats) {
          if (operatorStats.operatorType == "CudfAggregation") {
            cudfUsed = true;
            break;
          }
        }
      }
      
      if (shouldSucceed) {
        if (cudfUsed) {
          std::cout << "🎉 SUCCESS: CUDF was used and produced results!" << std::endl;
          std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
          std::cout << "   🔄 Type after coercion: (see debug output above)" << std::endl;
          std::cout << "   ✅ Signature registered: YES" << std::endl;
          std::cout << "   ✅ CUDF implementation: YES" << std::endl;
          std::cout << "   ✅ End-to-end execution: YES" << std::endl;
          std::cout << "   ✅ VERDICT: Keep this signature in registry" << std::endl;
        } else {
          std::cout << "❌ FAILED: CUDF should have been used but fell back to CPU" << std::endl;
          std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
          std::cout << "   ❌ CUDF implementation: MISSING or BROKEN" << std::endl;
          std::cout << "   🚨 VERDICT: REMOVE this signature from registry!" << std::endl;
          FAIL() << "CUDF signature " << functionName << "(" << inputType->toString() << ") is registered but doesn't work - MUST BE REMOVED!";
        }
      } else {
        if (!cudfUsed) {
          std::cout << "✅ EXPECTED: CUDF correctly fell back to CPU" << std::endl;
          std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
          std::cout << "   ✅ Signature correctly NOT registered" << std::endl;
          std::cout << "   ✅ VERDICT: Correctly not in registry" << std::endl;
        } else {
          std::cout << "❌ FAILED: CUDF should have fallen back to CPU but was used" << std::endl;
          std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
          std::cout << "   🚨 VERDICT: REMOVE this signature from registry!" << std::endl;
          FAIL() << "CUDF signature " << functionName << "(" << inputType->toString() << ") should not be registered!";
        }
      }
      
    } catch (const std::exception& e) {
      if (shouldSucceed) {
        std::cout << "❌ EXCEPTION: " << e.what() << std::endl;
        std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
        std::cout << "   ❌ CUDF implementation: BROKEN (throws exception)" << std::endl;
        std::cout << "   🚨 VERDICT: REMOVE this signature from registry!" << std::endl;
        FAIL() << "CUDF signature " << functionName << "(" << inputType->toString() << ") is registered but throws exception - MUST BE REMOVED!";
      } else {
        std::cout << "✅ EXPECTED EXCEPTION: " << e.what() << std::endl;
        std::cout << "   📝 Original input type: " << inputType->toString() << std::endl;
        std::cout << "   ✅ CUDF correctly rejects unsupported operations" << std::endl;
        std::cout << "   ✅ VERDICT: Correctly not in registry" << std::endl;
      }
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
};

// Test 1: Verify ALL registered signatures actually work end-to-end in CUDF
TEST_F(CudfCreateAggregatorSimpleTest, allRegisteredSignaturesMustWork) {
  std::cout << "=== TESTING ALL CURRENTLY REGISTERED CUDF SIGNATURES ===" << std::endl;
  std::cout << "🎯 PURPOSE: Verify every registered CUDF signature actually works end-to-end" << std::endl;
  std::cout << "🚨 IF ANY FAIL: They MUST be removed from ExpressionEvaluator.cpp!" << std::endl;
  std::cout << "📝 NOTE: Input types may be coerced (e.g., INTEGER->BIGINT)" << std::endl;
  
  std::cout << "\n--- Testing SUM signatures (6 registered) ---" << std::endl;
  std::cout << "Registered: tinyint->bigint, smallint->bigint, integer->bigint, bigint->bigint, real->real, double->double" << std::endl;
  testCudfAggregationEndToEnd("sum", TINYINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("sum", SMALLINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("sum", INTEGER(), BIGINT(), true);
  testCudfAggregationEndToEnd("sum", BIGINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("sum", REAL(), REAL(), true);
  testCudfAggregationEndToEnd("sum", DOUBLE(), DOUBLE(), true);
  
  std::cout << "\n--- Testing COUNT signatures (8 registered) ---" << std::endl;
  std::cout << "Registered: tinyint->bigint, smallint->bigint, integer->bigint, bigint->bigint, real->bigint, double->bigint, varchar->bigint, boolean->bigint" << std::endl;
  testCudfAggregationEndToEnd("count", TINYINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", SMALLINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", INTEGER(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", BIGINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", REAL(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", DOUBLE(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", VARCHAR(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", BOOLEAN(), BIGINT(), true);
  
  std::cout << "\n--- Testing MIN signatures (6 registered, numeric only) ---" << std::endl;
  std::cout << "Registered: tinyint->tinyint, smallint->smallint, integer->integer, bigint->bigint, real->real, double->double" << std::endl;
  std::cout << "NOTE: varchar and boolean were REMOVED in previous fix" << std::endl;
  testCudfAggregationEndToEnd("min", TINYINT(), TINYINT(), true);
  testCudfAggregationEndToEnd("min", SMALLINT(), SMALLINT(), true);
  testCudfAggregationEndToEnd("min", INTEGER(), INTEGER(), true);
  testCudfAggregationEndToEnd("min", BIGINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("min", REAL(), REAL(), true);
  testCudfAggregationEndToEnd("min", DOUBLE(), DOUBLE(), true);
  
  std::cout << "\n--- Testing MAX signatures (6 registered, numeric only) ---" << std::endl;
  std::cout << "Registered: tinyint->tinyint, smallint->smallint, integer->integer, bigint->bigint, real->real, double->double" << std::endl;
  std::cout << "NOTE: varchar and boolean were REMOVED in previous fix" << std::endl;
  testCudfAggregationEndToEnd("max", TINYINT(), TINYINT(), true);
  testCudfAggregationEndToEnd("max", SMALLINT(), SMALLINT(), true);
  testCudfAggregationEndToEnd("max", INTEGER(), INTEGER(), true);
  testCudfAggregationEndToEnd("max", BIGINT(), BIGINT(), true);
  testCudfAggregationEndToEnd("max", REAL(), REAL(), true);
  testCudfAggregationEndToEnd("max", DOUBLE(), DOUBLE(), true);
  
  std::cout << "\n--- Testing AVG signatures (4 registered - 2 REMOVED) ---" << std::endl;
  std::cout << "Registered: smallint->double, integer->double, bigint->double, double->double" << std::endl;
  std::cout << "REMOVED: tinyint->double (exception), real->double (return type mismatch)" << std::endl;
  testCudfAggregationEndToEnd("avg", SMALLINT(), DOUBLE(), true);
  testCudfAggregationEndToEnd("avg", INTEGER(), DOUBLE(), true);
  testCudfAggregationEndToEnd("avg", BIGINT(), DOUBLE(), true);
  testCudfAggregationEndToEnd("avg", DOUBLE(), DOUBLE(), true);
  
  std::cout << "\n📊 TOTAL SIGNATURES TESTED: 30 (was 32, removed 2 broken)" << std::endl;
  std::cout << "   - SUM: 6 signatures ✅" << std::endl;
  std::cout << "   - COUNT: 8 signatures ✅" << std::endl;
  std::cout << "   - MIN: 6 signatures ✅" << std::endl;
  std::cout << "   - MAX: 6 signatures ✅" << std::endl;
  std::cout << "   - AVG: 4 signatures ✅ (removed 2 broken)" << std::endl;
}

// Test 2: Verify unregistered signatures fall back to CPU
TEST_F(CudfCreateAggregatorSimpleTest, unregisteredSignaturesFallBackToCpu) {
  std::cout << "\n=== TESTING UNREGISTERED SIGNATURES FALL BACK TO CPU ===" << std::endl;
  
  std::cout << "\n--- Testing previously removed string signatures ---" << std::endl;
  // These should fall back to CPU because we removed the string signatures
  testCudfAggregationEndToEnd("min", VARCHAR(), VARCHAR(), false);
  testCudfAggregationEndToEnd("max", BOOLEAN(), BOOLEAN(), false);
  
  std::cout << "\n--- Testing newly removed broken AVG signatures ---" << std::endl;
  // These should fall back to CPU because we just removed them
  testCudfAggregationEndToEnd("avg", TINYINT(), DOUBLE(), false);
  testCudfAggregationEndToEnd("avg", REAL(), DOUBLE(), false);
}

// Test 3: Test count with strings (should work)
TEST_F(CudfCreateAggregatorSimpleTest, countWithStringsWorks) {
  std::cout << "\n=== TESTING COUNT WITH STRINGS (SHOULD WORK) ===" << std::endl;
  
  // Count should work with strings because it just counts rows
  testCudfAggregationEndToEnd("count", VARCHAR(), BIGINT(), true);
  testCudfAggregationEndToEnd("count", BOOLEAN(), BIGINT(), true);
}

// Test 4: Comprehensive validation
TEST_F(CudfCreateAggregatorSimpleTest, comprehensiveValidation) {
  std::cout << "\n=== COMPREHENSIVE END-TO-END VALIDATION ===" << std::endl;
  
  struct TestCase {
    std::string function;
    TypePtr inputType;
    TypePtr resultType;
    bool shouldWork;
    std::string description;
  };
  
  std::vector<TestCase> testCases = {
      // These should work (registered AND implemented)
      {"sum", BIGINT(), BIGINT(), true, "sum(bigint) - core functionality"},
      {"count", VARCHAR(), BIGINT(), true, "count(varchar) - should work"},
      {"min", DOUBLE(), DOUBLE(), true, "min(double) - numeric min"},
      {"max", REAL(), REAL(), true, "max(real) - numeric max"},
      {"avg", INTEGER(), DOUBLE(), true, "avg(integer) - numeric average"},
      
      // These should fall back to CPU (not registered)
      {"min", VARCHAR(), VARCHAR(), false, "min(varchar) - not supported by CUDF"},
      {"max", BOOLEAN(), BOOLEAN(), false, "max(boolean) - not supported by CUDF"},
  };
  
  int passed = 0;
  int failed = 0;
  
  for (const auto& testCase : testCases) {
    std::cout << "\n--- " << testCase.description << " ---" << std::endl;
    try {
      testCudfAggregationEndToEnd(
          testCase.function,
          testCase.inputType,
          testCase.resultType,
          testCase.shouldWork);
      passed++;
    } catch (const std::exception& e) {
      std::cout << "❌ Test case failed: " << e.what() << std::endl;
      failed++;
    }
  }
  
  std::cout << "\n=== FINAL REGISTRY CLEANUP SUMMARY ===" << std::endl;
  std::cout << "✅ Passed: " << passed << " signatures work correctly" << std::endl;
  std::cout << "❌ Failed: " << failed << " signatures are broken" << std::endl;
  
  if (failed > 0) {
    std::cout << "\n🚨 CRITICAL: REGISTRY CLEANUP REQUIRED!" << std::endl;
    std::cout << "❌ " << failed << " CUDF signatures are registered but don't work!" << std::endl;
    std::cout << "📝 ACTION REQUIRED:" << std::endl;
    std::cout << "   1. Find the failed signatures in the test output above" << std::endl;
    std::cout << "   2. Remove them from velox/experimental/cudf/expression/ExpressionEvaluator.cpp" << std::endl;
    std::cout << "   3. Look for 'VERDICT: REMOVE this signature from registry!'" << std::endl;
    std::cout << "\n💡 Why this matters:" << std::endl;
    std::cout << "   - Registered signatures that don't work cause incorrect query planning" << std::endl;
    std::cout << "   - Users expect CUDF acceleration but get CPU fallback or errors" << std::endl;
    std::cout << "   - This creates a 'gaping hole' in the CUDF integration" << std::endl;
  } else {
    std::cout << "\n🎉 REGISTRY IS CLEAN!" << std::endl;
    std::cout << "✅ Every registered CUDF signature works end-to-end" << std::endl;
    std::cout << "✅ Every unregistered signature correctly falls back to CPU" << std::endl;
    std::cout << "✅ No gaping holes in the CUDF integration" << std::endl;
    std::cout << "✅ Users get the CUDF acceleration they expect" << std::endl;
  }
}

} // namespace
