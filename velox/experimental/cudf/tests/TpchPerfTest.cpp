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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"

#include <gflags/gflags.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

DEFINE_int32(tpch_perf_lineitem_rows, 10000, "Lineitem rows to generate.");
DEFINE_int32(tpch_perf_part_rows, 2000, "Part rows to generate.");
DEFINE_int32(tpch_perf_batches, 1, "Number of batches to split data.");
DEFINE_int32(tpch_perf_iterations, 1, "Iterations per test.");

int32_t toDays(const std::string& date) {
  return DATE()->toDays(date);
}

std::string brandForKey(int64_t partKey) {
  switch (partKey % 3) {
    case 0:
      return "Brand#12";
    case 1:
      return "Brand#23";
    default:
      return "Brand#34";
  }
}

std::string containerForKey(int64_t partKey) {
  switch (partKey % 4) {
    case 0:
      return "SM CASE";
    case 1:
      return "MED BOX";
    case 2:
      return "LG PACK";
    default:
      return "SM PKG";
  }
}

} // namespace

class CudfTpchPerfTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    CudfConfig::getInstance().allowCpuFallback = false;
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA device not available";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(0)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> makeQ6LineitemDoubleBatches(
      size_t totalRows,
      size_t batches) {
    std::vector<RowVectorPtr> result;
    if (batches == 0) {
      return result;
    }
    const auto base = totalRows / batches;
    const auto rem = totalRows % batches;
    size_t offset = 0;
    auto start1993 = toDays("1993-01-01");
    for (size_t b = 0; b < batches; ++b) {
      const auto size = base + (b < rem ? 1 : 0);
      auto shipdate = vectorMaker_.flatVector<int32_t>(
          size,
          [&](auto row) {
            return start1993 + static_cast<int32_t>((offset + row) % 730);
          },
          nullptr,
          DATE());
      auto quantity = vectorMaker_.flatVector<double>(
          size, [&](auto row) { return 1.0 + ((offset + row) % 50); });
      auto discount = vectorMaker_.flatVector<double>(
          size, [&](auto row) { return ((offset + row) % 11) * 0.01; });
      auto extendedprice = vectorMaker_.flatVector<double>(
          size, [&](auto row) { return 1000.0 + ((offset + row) % 100); });
      result.push_back(makeRowVector(
          {"l_shipdate", "l_extendedprice", "l_quantity", "l_discount"},
          {shipdate, extendedprice, quantity, discount}));
      offset += size;
    }
    return result;
  }

  std::vector<RowVectorPtr> makeQ6LineitemDecimalBatches(
      size_t totalRows,
      size_t batches) {
    std::vector<RowVectorPtr> result;
    if (batches == 0) {
      return result;
    }
    const auto base = totalRows / batches;
    const auto rem = totalRows % batches;
    size_t offset = 0;
    auto start1993 = toDays("1993-01-01");
    for (size_t b = 0; b < batches; ++b) {
      const auto size = base + (b < rem ? 1 : 0);
      auto shipdate = vectorMaker_.flatVector<int32_t>(
          size,
          [&](auto row) {
            return start1993 + static_cast<int32_t>((offset + row) % 730);
          },
          nullptr,
          DATE());
      auto quantity = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            return static_cast<int64_t>(1 + ((offset + row) % 50)) * 100;
          },
          nullptr,
          DECIMAL(12, 2));
      auto discount = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) { return static_cast<int64_t>((offset + row) % 11); },
          nullptr,
          DECIMAL(5, 2));
      auto extendedprice = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            return static_cast<int64_t>(1000 + ((offset + row) % 100)) * 100;
          },
          nullptr,
          DECIMAL(12, 2));
      result.push_back(makeRowVector(
          {"l_shipdate", "l_extendedprice", "l_quantity", "l_discount"},
          {shipdate, extendedprice, quantity, discount}));
      offset += size;
    }
    return result;
  }

  RowVectorPtr makeQ19Part(size_t size) {
    auto partkey = vectorMaker_.flatVector<int64_t>(
        size, [&](auto row) { return static_cast<int64_t>(row + 1); });
    auto brand = vectorMaker_.flatVector<std::string>(
        size, [&](auto row) { return brandForKey(row + 1); });
    auto container = vectorMaker_.flatVector<std::string>(
        size, [&](auto row) { return containerForKey(row + 1); });
    auto sizeCol = vectorMaker_.flatVector<int32_t>(
        size,
        [&](auto row) {
          switch ((row + 1) % 3) {
            case 0:
              return 1 + static_cast<int32_t>((row + 1) % 5);
            case 1:
              return 1 + static_cast<int32_t>((row + 1) % 10);
            default:
              return 1 + static_cast<int32_t>((row + 1) % 15);
          }
        });
    return makeRowVector(
        {"p_partkey", "p_brand", "p_container", "p_size"},
        {partkey, brand, container, sizeCol});
  }

  std::vector<RowVectorPtr> makeQ19LineitemDoubleBatches(
      size_t totalRows,
      size_t batches,
      size_t partRows) {
    std::vector<RowVectorPtr> result;
    if (batches == 0) {
      return result;
    }
    const auto base = totalRows / batches;
    const auto rem = totalRows % batches;
    size_t offset = 0;
    for (size_t b = 0; b < batches; ++b) {
      const auto size = base + (b < rem ? 1 : 0);
      auto partkey = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            return static_cast<int64_t>((offset + row) % partRows) + 1;
          });
      auto quantity = vectorMaker_.flatVector<double>(
          size,
          [&](auto row) {
            auto key = static_cast<int64_t>((offset + row) % partRows) + 1;
            switch (key % 3) {
              case 0:
                return 1.0 + ((offset + row) % 11);
              case 1:
                return 10.0 + ((offset + row) % 11);
              default:
                return 20.0 + ((offset + row) % 11);
            }
          });
      auto extendedprice = vectorMaker_.flatVector<double>(
          size, [&](auto row) { return 1000.0 + ((offset + row) % 100); });
      auto discount = vectorMaker_.flatVector<double>(
          size, [&](auto row) { return ((offset + row) % 11) * 0.01; });
      auto shipmode = vectorMaker_.flatVector<std::string>(
          size,
          [&](auto row) {
            return ((offset + row) % 5 == 0) ? "TRUCK"
                                             : (((offset + row) % 2 == 0)
                                                    ? "AIR"
                                                    : "AIR REG");
          });
      auto shipinstruct = vectorMaker_.flatVector<std::string>(
          size,
          [&](auto row) {
            return ((offset + row) % 5 == 0) ? "NONE"
                                             : "DELIVER IN PERSON";
          });
      result.push_back(makeRowVector(
          {"l_partkey",
           "l_shipmode",
           "l_shipinstruct",
           "l_extendedprice",
           "l_discount",
           "l_quantity"},
          {partkey,
           shipmode,
           shipinstruct,
           extendedprice,
           discount,
           quantity}));
      offset += size;
    }
    return result;
  }

  std::vector<RowVectorPtr> makeQ19LineitemDecimalBatches(
      size_t totalRows,
      size_t batches,
      size_t partRows) {
    std::vector<RowVectorPtr> result;
    if (batches == 0) {
      return result;
    }
    const auto base = totalRows / batches;
    const auto rem = totalRows % batches;
    size_t offset = 0;
    for (size_t b = 0; b < batches; ++b) {
      const auto size = base + (b < rem ? 1 : 0);
      auto partkey = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            return static_cast<int64_t>((offset + row) % partRows) + 1;
          });
      auto quantity = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            auto key = static_cast<int64_t>((offset + row) % partRows) + 1;
            int64_t baseVal;
            switch (key % 3) {
              case 0:
                baseVal = 1 + ((offset + row) % 11);
                break;
              case 1:
                baseVal = 10 + ((offset + row) % 11);
                break;
              default:
                baseVal = 20 + ((offset + row) % 11);
                break;
            }
            return baseVal * 100;
          },
          nullptr,
          DECIMAL(12, 2));
      auto extendedprice = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) {
            return static_cast<int64_t>(1000 + ((offset + row) % 100)) * 100;
          },
          nullptr,
          DECIMAL(12, 2));
      auto discount = vectorMaker_.flatVector<int64_t>(
          size,
          [&](auto row) { return static_cast<int64_t>((offset + row) % 11); },
          nullptr,
          DECIMAL(5, 2));
      auto shipmode = vectorMaker_.flatVector<std::string>(
          size,
          [&](auto row) {
            return ((offset + row) % 5 == 0) ? "TRUCK"
                                             : (((offset + row) % 2 == 0)
                                                    ? "AIR"
                                                    : "AIR REG");
          });
      auto shipinstruct = vectorMaker_.flatVector<std::string>(
          size,
          [&](auto row) {
            return ((offset + row) % 5 == 0) ? "NONE"
                                             : "DELIVER IN PERSON";
          });
      result.push_back(makeRowVector(
          {"l_partkey",
           "l_shipmode",
           "l_shipinstruct",
           "l_extendedprice",
           "l_discount",
           "l_quantity"},
          {partkey,
           shipmode,
           shipinstruct,
           extendedprice,
           discount,
           quantity}));
      offset += size;
    }
    return result;
  }

  void runPlan(
      const core::PlanNodePtr& plan,
      const std::string& label) {
    for (int i = 0; i < FLAGS_tpch_perf_iterations; ++i) {
      auto start = std::chrono::steady_clock::now();
      auto result =
          exec::test::AssertQueryBuilder(plan).copyResults(pool());
      auto end = std::chrono::steady_clock::now();
      auto ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      LOG(INFO) << label << " iteration " << i << ": " << ms << " ms";
      VELOX_CHECK_NOT_NULL(result);
    }
  }
};

TEST_F(CudfTpchPerfTest, q6Double) {
  auto lineitem = makeQ6LineitemDoubleBatches(
      FLAGS_tpch_perf_lineitem_rows, FLAGS_tpch_perf_batches);
  const std::string filter =
      "l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND "
      "l_discount between 0.05 and 0.07 AND l_quantity < 24.0";
  auto plan = exec::test::PlanBuilder()
                  .values(lineitem)
                  .filter(filter)
                  .project({"l_extendedprice * l_discount AS revenue_part"})
                  .partialAggregation({}, {"sum(revenue_part) AS revenue"})
                  .finalAggregation()
                  .planNode();
  runPlan(plan, "q6_double");
}

TEST_F(CudfTpchPerfTest, q6Decimal) {
  auto lineitem = makeQ6LineitemDecimalBatches(
      FLAGS_tpch_perf_lineitem_rows, FLAGS_tpch_perf_batches);
  const std::string filter =
      "l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND "
      "l_discount between CAST('0.05' AS DECIMAL(5, 2)) AND "
      "CAST('0.07' AS DECIMAL(5, 2)) AND "
      "l_quantity < CAST('24.00' AS DECIMAL(12, 2))";
  auto plan = exec::test::PlanBuilder()
                  .values(lineitem)
                  .filter(filter)
                  .project({"l_extendedprice * l_discount AS revenue_part"})
                  .partialAggregation({}, {"sum(revenue_part) AS revenue"})
                  .finalAggregation()
                  .planNode();
  runPlan(plan, "q6_decimal");
}

TEST_F(CudfTpchPerfTest, q19Double) {
  auto part = makeQ19Part(FLAGS_tpch_perf_part_rows);
  auto lineitem = makeQ19LineitemDoubleBatches(
      FLAGS_tpch_perf_lineitem_rows,
      FLAGS_tpch_perf_batches,
      FLAGS_tpch_perf_part_rows);

  const std::string lineitemFilter =
      "l_shipmode IN ('AIR', 'AIR REG') AND "
      "l_shipinstruct = 'DELIVER IN PERSON'";
  const std::string joinFilter =
      "     ((p_brand = 'Brand#12')"
      "     AND (l_quantity between 1.0 and 11.0)"
      "     AND (p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))"
      "     AND (p_size BETWEEN 1 AND 5))"
      " OR  ((p_brand ='Brand#23')"
      "     AND (p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))"
      "     AND (l_quantity between 10.0 and 20.0)"
      "     AND (p_size BETWEEN 1 AND 10))"
      " OR  ((p_brand = 'Brand#34')"
      "     AND (p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))"
      "     AND (l_quantity between 20.0 and 30.0)"
      "     AND (p_size BETWEEN 1 AND 15))";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto partPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool())
                      .values({part})
                      .planNode();
  auto plan =
      exec::test::PlanBuilder(planNodeIdGenerator, pool())
          .values(lineitem)
          .filter(lineitemFilter)
          .project({"l_extendedprice * (1.0 - l_discount) AS part_revenue",
                    "l_partkey",
                    "l_quantity"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partPlan,
              joinFilter,
              {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) AS revenue"})
          .finalAggregation()
          .planNode();

  runPlan(plan, "q19_double");
}

TEST_F(CudfTpchPerfTest, q19Decimal) {
  auto part = makeQ19Part(FLAGS_tpch_perf_part_rows);
  auto lineitem = makeQ19LineitemDecimalBatches(
      FLAGS_tpch_perf_lineitem_rows,
      FLAGS_tpch_perf_batches,
      FLAGS_tpch_perf_part_rows);

  const std::string lineitemFilter =
      "l_shipmode IN ('AIR', 'AIR REG') AND "
      "l_shipinstruct = 'DELIVER IN PERSON'";
  const std::string joinFilter =
      "     ((p_brand = 'Brand#12')"
      "     AND (l_quantity between "
      "CAST('1.00' AS DECIMAL(12, 2)) AND CAST('11.00' AS DECIMAL(12, 2)))"
      "     AND (p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))"
      "     AND (p_size BETWEEN 1 AND 5))"
      " OR  ((p_brand ='Brand#23')"
      "     AND (p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))"
      "     AND (l_quantity between "
      "CAST('10.00' AS DECIMAL(12, 2)) AND CAST('20.00' AS DECIMAL(12, 2)))"
      "     AND (p_size BETWEEN 1 AND 10))"
      " OR  ((p_brand = 'Brand#34')"
      "     AND (p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))"
      "     AND (l_quantity between "
      "CAST('20.00' AS DECIMAL(12, 2)) AND CAST('30.00' AS DECIMAL(12, 2)))"
      "     AND (p_size BETWEEN 1 AND 15))";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto partPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool())
                      .values({part})
                      .planNode();
  auto plan =
      exec::test::PlanBuilder(planNodeIdGenerator, pool())
          .values(lineitem)
          .filter(lineitemFilter)
          .project(
              {"l_extendedprice * "
               "(CAST('1.00' AS DECIMAL(5, 2)) - l_discount) AS part_revenue",
               "l_partkey",
               "l_quantity"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partPlan,
              joinFilter,
              {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) AS revenue"})
          .finalAggregation()
          .planNode();

  runPlan(plan, "q19_decimal");
}

} // namespace facebook::velox::cudf_velox
