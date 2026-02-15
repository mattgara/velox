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
#include <vector>

namespace {

struct Options {
  int64_t numKeys{19'412'603};
  int64_t startKey{1};
  int32_t scale{2};
  int32_t decimalBits{128};
  int32_t rowsPerKey{1};
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
    if (opts.numKeys > std::numeric_limits<int64_t>::max() / opts.rowsPerKey) {
      throw std::runtime_error("num-keys * rows-per-key overflow");
    }

    std::cout << "[DecimalGroupbyRepro] numKeys=" << opts.numKeys
              << " startKey=" << opts.startKey << " scale=" << opts.scale
              << " decimalBits=" << opts.decimalBits
              << " rowsPerKey=" << opts.rowsPerKey
              << " valueMode=" << opts.valueMode
              << " modValue=" << opts.modValue << " validate=" << opts.validate
              << std::endl;

    auto stream = cudf::get_default_stream();
    const auto totalRows = opts.numKeys * opts.rowsPerKey;
    if (totalRows > std::numeric_limits<cudf::size_type>::max()) {
      throw std::runtime_error("num-keys * rows-per-key exceeds cudf::size_type");
    }
    const auto numRows = static_cast<cudf::size_type>(totalRows);
    const auto valueScale = numeric::scale_type{-opts.scale};
    const auto decimalType = cudf::data_type{
        opts.decimalBits == 128 ? cudf::type_id::DECIMAL128
                                : cudf::type_id::DECIMAL64,
        valueScale};

    std::vector<int64_t> hostKeys(numRows);
    if (opts.decimalBits == 128) {
      std::vector<__int128_t> hostValues(numRows);
      for (int64_t i = 0; i < opts.numKeys; ++i) {
        int64_t key = opts.startKey + i;
        __int128_t value = buildValue128(key, opts);
        for (int32_t r = 0; r < opts.rowsPerKey; ++r) {
          auto idx = static_cast<size_t>(i * opts.rowsPerKey + r);
          hostKeys[idx] = key;
          hostValues[idx] = value;
        }
      }

      auto keyCol = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT64},
          numRows,
          cudf::mask_state::UNALLOCATED,
          stream);
      auto valCol = cudf::make_fixed_width_column(
          decimalType,
          numRows,
          cudf::mask_state::UNALLOCATED,
          stream);

      auto keyBytes = static_cast<size_t>(numRows) * sizeof(int64_t);
      auto valBytes = static_cast<size_t>(numRows) * sizeof(__int128_t);
      cudaMemcpyAsync(
          keyCol->mutable_view().data<int64_t>(),
          hostKeys.data(),
          keyBytes,
          cudaMemcpyHostToDevice,
          stream.value());
      cudaMemcpyAsync(
          valCol->mutable_view().data<__int128_t>(),
          hostValues.data(),
          valBytes,
          cudaMemcpyHostToDevice,
          stream.value());

      cudf::table_view keyTable({keyCol->view()});
      cudf::groupby::groupby groupby(keyTable, cudf::null_policy::EXCLUDE);
      cudf::groupby::aggregation_request request;
      request.values = valCol->view();
      request.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      auto output = groupby.aggregate({request}, stream);
      stream.synchronize();

      auto outKeysView = output.first->get_column(0).view();
      auto outValsView = output.second[0].results[0]->view();
      const auto outRows = outKeysView.size();
      std::vector<int64_t> outKeys(outRows);

      bool outIs128 = outValsView.type().id() == cudf::type_id::DECIMAL128;
      std::vector<__int128_t> outVals128(outRows);
      std::vector<int64_t> outVals64(outRows);

      cudaMemcpyAsync(
          outKeys.data(),
          outKeysView.data<int64_t>(),
          static_cast<size_t>(outRows) * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.value());
      if (outIs128) {
        cudaMemcpyAsync(
            outVals128.data(),
            outValsView.data<__int128_t>(),
            static_cast<size_t>(outRows) * sizeof(__int128_t),
            cudaMemcpyDeviceToHost,
            stream.value());
      } else {
        cudaMemcpyAsync(
            outVals64.data(),
            outValsView.data<int64_t>(),
            static_cast<size_t>(outRows) * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.value());
      }
      stream.synchronize();

      std::cout << "[DecimalGroupbyRepro] outputRows=" << outRows << std::endl;
      if (outRows != opts.numKeys) {
        std::cout << "[DecimalGroupbyRepro] warning: outputRows "
                  << outRows << " != numKeys " << opts.numKeys << std::endl;
      }
      if (!opts.validate) {
        return 0;
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
      return mismatches > 0 ? 2 : 0;
    }

    std::vector<int64_t> hostValues(numRows);
    for (int64_t i = 0; i < opts.numKeys; ++i) {
      int64_t key = opts.startKey + i;
      __int128_t value128 = buildValue128(key, opts);
      if (value128 > std::numeric_limits<int64_t>::max() ||
          value128 < std::numeric_limits<int64_t>::min()) {
        throw std::runtime_error("Value out of int64 range for DECIMAL64");
      }
      auto value = static_cast<int64_t>(value128);
      for (int32_t r = 0; r < opts.rowsPerKey; ++r) {
        auto idx = static_cast<size_t>(i * opts.rowsPerKey + r);
        hostKeys[idx] = key;
        hostValues[idx] = value;
      }
    }

    auto keyCol = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        numRows,
        cudf::mask_state::UNALLOCATED,
        stream);
    auto valCol = cudf::make_fixed_width_column(
        decimalType,
        numRows,
        cudf::mask_state::UNALLOCATED,
        stream);
    auto keyBytes = static_cast<size_t>(numRows) * sizeof(int64_t);
    auto valBytes = static_cast<size_t>(numRows) * sizeof(int64_t);
    cudaMemcpyAsync(
        keyCol->mutable_view().data<int64_t>(),
        hostKeys.data(),
        keyBytes,
        cudaMemcpyHostToDevice,
        stream.value());
    cudaMemcpyAsync(
        valCol->mutable_view().data<int64_t>(),
        hostValues.data(),
        valBytes,
        cudaMemcpyHostToDevice,
        stream.value());

    cudf::table_view keyTable({keyCol->view()});
    cudf::groupby::groupby groupby(keyTable, cudf::null_policy::EXCLUDE);
    cudf::groupby::aggregation_request request;
    request.values = valCol->view();
    request.aggregations.push_back(
        cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    auto output = groupby.aggregate({request}, stream);
    stream.synchronize();

    auto outKeysView = output.first->get_column(0).view();
    auto outValsView = output.second[0].results[0]->view();
    const auto outRows = outKeysView.size();
    std::vector<int64_t> outKeys(outRows);
    bool outIs128 = outValsView.type().id() == cudf::type_id::DECIMAL128;
    std::vector<__int128_t> outVals128(outRows);
    std::vector<int64_t> outVals64(outRows);

    cudaMemcpyAsync(
        outKeys.data(),
        outKeysView.data<int64_t>(),
        static_cast<size_t>(outRows) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value());
    if (outIs128) {
      cudaMemcpyAsync(
          outVals128.data(),
          outValsView.data<__int128_t>(),
          static_cast<size_t>(outRows) * sizeof(__int128_t),
          cudaMemcpyDeviceToHost,
          stream.value());
    } else {
      cudaMemcpyAsync(
          outVals64.data(),
          outValsView.data<int64_t>(),
          static_cast<size_t>(outRows) * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.value());
    }
    stream.synchronize();

    std::cout << "[DecimalGroupbyRepro] outputRows=" << outRows << std::endl;
    if (outRows != opts.numKeys) {
      std::cout << "[DecimalGroupbyRepro] warning: outputRows "
                << outRows << " != numKeys " << opts.numKeys << std::endl;
    }
    if (!opts.validate) {
      return 0;
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
    return mismatches > 0 ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "[DecimalGroupbyRepro] error: " << e.what() << std::endl;
    return 1;
  }
}
