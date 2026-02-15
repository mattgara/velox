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

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Options {
  int64_t numKeys{19'412'603};
  int64_t startKey{1};
  int32_t scale{2};
  int32_t decimalBits{128};
  int32_t rowsPerKey{1};
  int64_t batchRows{0};
  std::string rowOrder{"by-key"};
  std::string mergeMode{"auto"};
  std::string valueMode{"key"};
  int64_t modValue{1000};
  bool validate{true};
};

void printUsage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --num-keys N        Number of unique keys (default 19412603)\n"
            << "  --start-key K       Starting key value (default 1)\n"
            << "  --scale S           Decimal scale (default 2)\n"
            << "  --decimal-bits B    Decimal bits: 64 or 128 (default 128)\n"
            << "  --rows-per-key N    Rows per key (default 1)\n"
            << "  --batch-rows N      Rows per batch (default all rows)\n"
            << "  --row-order MODE    by-key|round-robin (default by-key)\n"
            << "  --merge-mode MODE   auto|single|incremental|two-phase (default auto)\n"
            << "  --value-mode MODE   key|mod|constant (default key)\n"
            << "  --mod M             Modulus for value-mode=mod (default 1000)\n"
            << "  --no-validate       Skip output verification\n"
            << "  --help              Show this message\n";
}

Options parseArgs(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    auto requireValue = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--num-keys") {
      opts.numKeys = std::stoll(requireValue("--num-keys"));
    } else if (arg == "--start-key") {
      opts.startKey = std::stoll(requireValue("--start-key"));
    } else if (arg == "--scale") {
      opts.scale = static_cast<int32_t>(std::stoll(requireValue("--scale")));
    } else if (arg == "--decimal-bits") {
      opts.decimalBits =
          static_cast<int32_t>(std::stoll(requireValue("--decimal-bits")));
    } else if (arg == "--rows-per-key") {
      opts.rowsPerKey =
          static_cast<int32_t>(std::stoll(requireValue("--rows-per-key")));
    } else if (arg == "--batch-rows") {
      opts.batchRows = std::stoll(requireValue("--batch-rows"));
    } else if (arg == "--row-order") {
      opts.rowOrder = requireValue("--row-order");
    } else if (arg == "--merge-mode") {
      opts.mergeMode = requireValue("--merge-mode");
    } else if (arg == "--value-mode") {
      opts.valueMode = requireValue("--value-mode");
    } else if (arg == "--mod") {
      opts.modValue = std::stoll(requireValue("--mod"));
    } else if (arg == "--no-validate") {
      opts.validate = false;
    } else if (arg == "--help") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return opts;
}

enum class RowOrder { kByKey, kRoundRobin };
enum class MergeMode { kSingle, kIncremental, kTwoPhase };

RowOrder parseRowOrder(const std::string& value) {
  if (value == "by-key") {
    return RowOrder::kByKey;
  }
  if (value == "round-robin") {
    return RowOrder::kRoundRobin;
  }
  throw std::runtime_error("row-order must be by-key or round-robin");
}

MergeMode parseMergeMode(const std::string& value) {
  if (value == "single") {
    return MergeMode::kSingle;
  }
  if (value == "incremental") {
    return MergeMode::kIncremental;
  }
  if (value == "two-phase") {
    return MergeMode::kTwoPhase;
  }
  throw std::runtime_error("merge-mode must be single, incremental, or two-phase");
}

const char* rowOrderName(RowOrder order) {
  return order == RowOrder::kRoundRobin ? "round-robin" : "by-key";
}

const char* mergeModeName(MergeMode mode) {
  switch (mode) {
    case MergeMode::kSingle:
      return "single";
    case MergeMode::kIncremental:
      return "incremental";
    case MergeMode::kTwoPhase:
      return "two-phase";
  }
  return "unknown";
}

int64_t keyForRow(int64_t row, const Options& opts, RowOrder order) {
  if (order == RowOrder::kByKey) {
    return opts.startKey + (row / static_cast<int64_t>(opts.rowsPerKey));
  }
  return opts.startKey + (row % opts.numKeys);
}

__int128_t pow10Int128(int32_t scale) {
  __int128_t value = 1;
  for (int32_t i = 0; i < scale; ++i) {
    value *= 10;
  }
  return value;
}

__int128_t buildValue128(int64_t key, const Options& opts) {
  const __int128_t scaleFactor = pow10Int128(opts.scale);
  if (opts.valueMode == "constant") {
    return scaleFactor;
  }
  if (opts.valueMode == "mod") {
    return static_cast<__int128_t>(key % opts.modValue) * scaleFactor;
  }
  return static_cast<__int128_t>(key) * scaleFactor;
}

std::string toString128(__int128_t value) {
  if (value == 0) {
    return "0";
  }
  bool negative = value < 0;
  __int128_t absValue = negative ? -value : value;
  std::string out;
  while (absValue > 0) {
    int digit = static_cast<int>(absValue % 10);
    out.push_back(static_cast<char>('0' + digit));
    absValue /= 10;
  }
  if (negative) {
    out.push_back('-');
  }
  std::reverse(out.begin(), out.end());
  return out;
}

template <typename T>
std::unique_ptr<cudf::table> buildInputTable(
    int64_t startRow,
    int64_t numRows,
    const Options& opts,
    RowOrder rowOrder,
    const cudf::data_type& decimalType,
    rmm::cuda_stream_view stream) {
  auto rows = static_cast<cudf::size_type>(numRows);
  std::vector<int64_t> hostKeys(rows);
  std::vector<T> hostValues(rows);
  for (int64_t i = 0; i < numRows; ++i) {
    auto idx = static_cast<size_t>(i);
    int64_t key = keyForRow(startRow + i, opts, rowOrder);
    __int128_t value128 = buildValue128(key, opts);
    if constexpr (std::is_same_v<T, int64_t>) {
      if (value128 > std::numeric_limits<int64_t>::max() ||
          value128 < std::numeric_limits<int64_t>::min()) {
        throw std::runtime_error("Value out of int64 range for DECIMAL64");
      }
    }
    hostKeys[idx] = key;
    hostValues[idx] = static_cast<T>(value128);
  }

  auto keyCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      rows,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto valCol = cudf::make_fixed_width_column(
      decimalType,
      rows,
      cudf::mask_state::UNALLOCATED,
      stream);

  auto keyBytes = static_cast<size_t>(rows) * sizeof(int64_t);
  auto valBytes = static_cast<size_t>(rows) * sizeof(T);
  cudaMemcpyAsync(
      keyCol->mutable_view().data<int64_t>(),
      hostKeys.data(),
      keyBytes,
      cudaMemcpyHostToDevice,
      stream.value());
  cudaMemcpyAsync(
      valCol->mutable_view().data<T>(),
      hostValues.data(),
      valBytes,
      cudaMemcpyHostToDevice,
      stream.value());

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(keyCol));
  columns.push_back(std::move(valCol));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> groupbySumTable(
    const cudf::table_view& input,
    rmm::cuda_stream_view stream) {
  cudf::table_view keyTable({input.column(0)});
  cudf::groupby::groupby groupby(keyTable, cudf::null_policy::EXCLUDE);
  cudf::groupby::aggregation_request request;
  request.values = input.column(1);
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.push_back(std::move(request));
  auto output = groupby.aggregate(requests, stream);
  stream.synchronize();

  auto keyCols = output.first->release();
  keyCols.push_back(std::move(output.second[0].results[0]));
  return std::make_unique<cudf::table>(std::move(keyCols));
}

template <typename T>
std::unique_ptr<cudf::table> runAggregation(
    const Options& opts,
    RowOrder rowOrder,
    MergeMode mergeMode,
    int64_t totalRows,
    int64_t batchRows,
    const cudf::data_type& decimalType,
    rmm::cuda_stream_view stream) {
  std::unique_ptr<cudf::table> output;
  std::vector<std::unique_ptr<cudf::table>> staged;

  for (int64_t start = 0; start < totalRows; start += batchRows) {
    int64_t rowsThis = std::min(batchRows, totalRows - start);
    auto input = buildInputTable<T>(
        start, rowsThis, opts, rowOrder, decimalType, stream);
    auto batchAgg = groupbySumTable(input->view(), stream);

    if (mergeMode == MergeMode::kSingle) {
      output = std::move(batchAgg);
      break;
    }
    if (mergeMode == MergeMode::kIncremental) {
      if (output) {
        std::vector<cudf::table_view> tables;
        tables.push_back(output->view());
        tables.push_back(batchAgg->view());
        auto concatenated = cudf::concatenate(tables, stream);
        output = groupbySumTable(concatenated->view(), stream);
      } else {
        output = std::move(batchAgg);
      }
    } else {
      staged.push_back(std::move(batchAgg));
    }
  }

  if (mergeMode == MergeMode::kTwoPhase) {
    if (staged.empty()) {
      return nullptr;
    }
    if (staged.size() == 1) {
      output = std::move(staged[0]);
    } else {
      std::vector<cudf::table_view> tables;
      tables.reserve(staged.size());
      for (const auto& tbl : staged) {
        tables.push_back(tbl->view());
      }
      auto concatenated = cudf::concatenate(tables, stream);
      output = groupbySumTable(concatenated->view(), stream);
    }
  }

  return output;
}

int64_t validateOutput(
    const cudf::table_view& output,
    const Options& opts,
    rmm::cuda_stream_view stream) {
  auto outKeysView = output.column(0);
  auto outValsView = output.column(1);
  const auto outRows = outKeysView.size();
  std::vector<int64_t> outKeys(outRows);

  bool outIs128 = outValsView.type().id() == cudf::type_id::DECIMAL128;
  std::vector<__int128_t> outVals128(outRows);
  std::vector<int64_t> outVals64(outRows);

  auto outKeysPtr = static_cast<const int64_t*>(outKeysView.head()) +
      outKeysView.offset();
  cudaMemcpyAsync(
      outKeys.data(),
      outKeysPtr,
      static_cast<size_t>(outRows) * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value());
  if (outIs128) {
    auto outValsPtr128 = static_cast<const __int128_t*>(outValsView.head()) +
        outValsView.offset();
    cudaMemcpyAsync(
        outVals128.data(),
        outValsPtr128,
        static_cast<size_t>(outRows) * sizeof(__int128_t),
        cudaMemcpyDeviceToHost,
        stream.value());
  } else {
    auto outValsPtr64 = static_cast<const int64_t*>(outValsView.head()) +
        outValsView.offset();
    cudaMemcpyAsync(
        outVals64.data(),
        outValsPtr64,
        static_cast<size_t>(outRows) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  stream.synchronize();

  std::cout << "[DecimalGroupbyRepro] outputRows=" << outRows << std::endl;
  if (outRows != opts.numKeys) {
    std::cout << "[DecimalGroupbyRepro] warning: outputRows " << outRows
              << " != numKeys " << opts.numKeys << std::endl;
  }

  std::vector<std::pair<int64_t, __int128_t>> pairs;
  pairs.reserve(outRows);
  for (cudf::size_type i = 0; i < outRows; ++i) {
    __int128_t value = outIs128 ? outVals128[i]
                                : static_cast<__int128_t>(outVals64[i]);
    pairs.emplace_back(outKeys[i], value);
  }
  std::sort(pairs.begin(), pairs.end());

  int64_t mismatches = 0;
  std::pair<int64_t, __int128_t> firstMismatch{0, 0};
  __int128_t firstExpected = 0;
  for (int64_t i = 0; i < opts.numKeys && i < outRows; ++i) {
    int64_t key = opts.startKey + i;
    __int128_t expected = buildValue128(key, opts) * opts.rowsPerKey;
    const auto& [outKey, outVal] = pairs[static_cast<size_t>(i)];
    if (outKey != key || outVal != expected) {
      if (mismatches == 0) {
        firstMismatch = {outKey, outVal};
        firstExpected = expected;
      }
      ++mismatches;
    }
  }

  std::cout << "[DecimalGroupbyRepro] mismatches=" << mismatches;
  if (mismatches > 0) {
    std::cout << " firstMismatchKey=" << firstMismatch.first
              << " outVal=" << toString128(firstMismatch.second)
              << " expected=" << toString128(firstExpected);
  }
  std::cout << std::endl;
  return mismatches;
}

} // namespace

int main(int argc, char** argv) {
  try {
    auto opts = parseArgs(argc, argv);
    if (opts.numKeys <= 0) {
      throw std::runtime_error("num-keys must be positive");
    }
    if (opts.rowsPerKey <= 0) {
      throw std::runtime_error("rows-per-key must be positive");
    }
    if (opts.scale < 0 || opts.scale > 18) {
      throw std::runtime_error("scale must be between 0 and 18");
    }
    if (opts.decimalBits != 64 && opts.decimalBits != 128) {
      throw std::runtime_error("decimal-bits must be 64 or 128");
    }
    if (opts.batchRows < 0) {
      throw std::runtime_error("batch-rows must be non-negative");
    }
    if (opts.numKeys > std::numeric_limits<int64_t>::max() / opts.rowsPerKey) {
      throw std::runtime_error("num-keys * rows-per-key overflow");
    }

    RowOrder rowOrder = parseRowOrder(opts.rowOrder);
    const auto totalRows = opts.numKeys * opts.rowsPerKey;
    int64_t batchRows = opts.batchRows > 0 ? opts.batchRows : totalRows;
    if (batchRows <= 0) {
      throw std::runtime_error("batch-rows must be positive");
    }
    if (batchRows > totalRows) {
      batchRows = totalRows;
    }

    std::string mergeModeStr = opts.mergeMode;
    if (mergeModeStr == "auto") {
      mergeModeStr = (batchRows < totalRows) ? "incremental" : "single";
    }
    MergeMode mergeMode = parseMergeMode(mergeModeStr);
    if (mergeMode == MergeMode::kSingle) {
      batchRows = totalRows;
    }

    if (batchRows > std::numeric_limits<cudf::size_type>::max()) {
      throw std::runtime_error("batch-rows exceeds cudf::size_type");
    }

    std::cout << "[DecimalGroupbyRepro] numKeys=" << opts.numKeys
              << " startKey=" << opts.startKey << " scale=" << opts.scale
              << " decimalBits=" << opts.decimalBits
              << " rowsPerKey=" << opts.rowsPerKey
              << " batchRows=" << batchRows
              << " rowOrder=" << rowOrderName(rowOrder)
              << " mergeMode=" << mergeModeName(mergeMode)
              << " valueMode=" << opts.valueMode
              << " modValue=" << opts.modValue << " validate=" << opts.validate
              << std::endl;

    auto stream = cudf::get_default_stream();
    const auto valueScale = numeric::scale_type{-opts.scale};
    const auto decimalType = cudf::data_type{
        opts.decimalBits == 128 ? cudf::type_id::DECIMAL128
                                : cudf::type_id::DECIMAL64,
        valueScale};

    std::unique_ptr<cudf::table> output;
    if (opts.decimalBits == 128) {
      output = runAggregation<__int128_t>(
          opts, rowOrder, mergeMode, totalRows, batchRows, decimalType, stream);
    }

    if (opts.decimalBits == 64) {
      output = runAggregation<int64_t>(
          opts, rowOrder, mergeMode, totalRows, batchRows, decimalType, stream);
    }

    if (!output) {
      throw std::runtime_error("No output produced");
    }
    if (!opts.validate) {
      return 0;
    }
    auto mismatches = validateOutput(output->view(), opts, stream);
    return mismatches > 0 ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "[DecimalGroupbyRepro] error: " << e.what() << std::endl;
    return 1;
  }
}
