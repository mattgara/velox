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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

namespace facebook::velox::cudf_velox {

cudf::type_id veloxToCudfTypeId(const TypePtr& type) {
  // DEBUG LOGGING: Log type conversion attempts
  std::cout << "=== veloxToCudfTypeId DEBUG ===" << std::endl;
  std::cout << "Converting type: " << type->toString() << std::endl;
  std::cout << "Type kind: " << TypeKindName::toName(type->kind()) << std::endl;
  
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      std::cout << "Mapped to: BOOL8" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::BOOL8;
    case TypeKind::TINYINT:
      std::cout << "Mapped to: INT8" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::INT8;
    case TypeKind::SMALLINT:
      std::cout << "Mapped to: INT16" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::INT16;
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        std::cout << "Mapped to: TIMESTAMP_DAYS (date)" << std::endl;
        std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
        return cudf::type_id::TIMESTAMP_DAYS;
      }
      std::cout << "Mapped to: INT32" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::INT32;
    case TypeKind::BIGINT:
      std::cout << "Mapped to: INT64" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::INT64;
    case TypeKind::REAL:
      std::cout << "Mapped to: FLOAT32" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::FLOAT32;
    case TypeKind::DOUBLE:
      std::cout << "Mapped to: FLOAT64" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::FLOAT64;
    case TypeKind::VARCHAR:
      std::cout << "Mapped to: STRING" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::STRING;
    case TypeKind::VARBINARY:
      std::cout << "Mapped to: STRING (varbinary)" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::STRING;
    case TypeKind::TIMESTAMP:
      std::cout << "Mapped to: TIMESTAMP_NANOSECONDS" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::TIMESTAMP_NANOSECONDS;
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    // TODO: Decimals are now logical types:
    // https://github.com/facebookincubator/velox/commit/73d2f935b55f084d30557c7be94b9768efb8e56f
    // case TypeKind::SHORT_DECIMAL: return cudf::type_id::DECIMAL64;
    // case TypeKind::LONG_DECIMAL: return cudf::type_id::DECIMAL128;
    case TypeKind::ARRAY:
      std::cout << "Mapped to: LIST" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::LIST;
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    case TypeKind::ROW:
      std::cout << "Mapped to: STRUCT" << std::endl;
      // For ROW types, also log the children
      if (type->isRow()) {
        auto rowType = std::dynamic_pointer_cast<const RowType>(type);
        std::cout << "ROW type children:" << std::endl;
        for (int i = 0; i < rowType->size(); ++i) {
          std::cout << "  Child " << i << " (" << rowType->nameOf(i) << "): " 
                    << rowType->childAt(i)->toString() << std::endl;
        }
      }
      std::cout << "=== END veloxToCudfTypeId DEBUG ===" << std::endl;
      return cudf::type_id::STRUCT;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      std::cout << "ERROR: Unsupported Velox type!" << std::endl;
      std::cout << "=== END veloxToCudfTypeId DEBUG (ERROR) ===" << std::endl;
      CUDF_FAIL(
          "Unsupported Velox type: " +
          std::string(TypeKindName::toName(type->kind())));
      return cudf::type_id::EMPTY;
  }
}

namespace with_arrow {

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  // DEBUG LOGGING: Log Arrow conversion details
  std::cout << "=== toCudfTable DEBUG ===" << std::endl;
  std::cout << "Input table type: " << veloxTable->type()->toString() << std::endl;
  std::cout << "Input table size: " << veloxTable->size() << std::endl;
  
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask.
  ArrowOptions arrowOptions{true, true};
  ArrowArray arrowArray;
  
  std::cout << "Exporting to Arrow array..." << std::endl;
  try {
    exportToArrow(
        std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
        arrowArray,
        pool,
        arrowOptions);
  } catch (const std::exception& e) {
    std::cout << "ERROR in exportToArrow (array): " << e.what() << std::endl;
    throw;
  }
  
  ArrowSchema arrowSchema;
  std::cout << "Exporting to Arrow schema..." << std::endl;
  try {
    exportToArrow(
        std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
        arrowSchema,
        arrowOptions);
  } catch (const std::exception& e) {
    std::cout << "ERROR in exportToArrow (schema): " << e.what() << std::endl;
    throw;
  }
  
  std::cout << "Converting from Arrow to cudf..." << std::endl;
  std::unique_ptr<cudf::table> tbl;
  try {
    tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream);
  } catch (const std::exception& e) {
    std::cout << "ERROR in cudf::from_arrow: " << e.what() << std::endl;
    std::cout << "=== END toCudfTable DEBUG (ERROR) ===" << std::endl;
    throw;
  }

  // Release Arrow resources
  if (arrowArray.release) {
    arrowArray.release(&arrowArray);
  }
  if (arrowSchema.release) {
    arrowSchema.release(&arrowSchema);
  }
  
  std::cout << "Successfully converted to cudf table" << std::endl;
  std::cout << "=== END toCudfTable DEBUG ===" << std::endl;
  return tbl;
}

namespace {

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    rmm::cuda_stream_view stream) {
  auto arrowDeviceArray = cudf::to_arrow_host(table, stream);
  auto& arrowArray = arrowDeviceArray->array;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  auto veloxTable = importFromArrowAsOwner(*arrowSchema, arrowArray, pool);
  // BaseVector to RowVector
  auto castedPtr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(castedPtr);
  return castedPtr;
}

template <typename Iterator>
std::vector<cudf::column_metadata>
getMetadata(Iterator begin, Iterator end, const std::string& namePrefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(namePrefix + std::to_string(i)));
    metadata.back().children_meta = getMetadata(
        c->child_begin(), c->child_end(), namePrefix + std::to_string(i));
    i++;
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, stream);
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return toVeloxColumn(table, pool, metadata, stream);
}

} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
