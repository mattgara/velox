# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_guard(GLOBAL)

# 3.30.4 is the minimum version required by cudf
cmake_minimum_required(VERSION 3.30.4)

# rapids_cmake commit 5ec2245 from 2026-01-26
set(VELOX_rapids_cmake_VERSION 26.04)
set(VELOX_rapids_cmake_COMMIT 5ec22457e58953e0a68f0745ce7a11a896ba62b1)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  bf7d4ed5885f5fe012c42fb0977e1fe1416896479ffd34baa0cf762d3e83dc80
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit e728b29 from 2026-01-26
set(VELOX_rmm_VERSION 26.04)
set(VELOX_rmm_COMMIT e728b2923f748d71aad30294b6926f43cb4c826e)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  ec18d881b327514de154af67a33a1288eec7bcd86909f23c9bf2d90511b0cf2f
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 0f03349 from 2026-01-26
set(VELOX_kvikio_VERSION 26.04)
set(VELOX_kvikio_COMMIT 0f03349bcaf029a2f582d9915a88d09e355ac691)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  728868c671e2686b5e9b7b4122d1661475f803c4fb98c0852d7be65c365d7b2d
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 68a0714 from 2026-01-27
set(VELOX_cudf_VERSION 26.04 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 68a0714a3701431041cb47bf1163706f597f9f48)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  0c723d7fd04eab60336dd4bcce41e225821d13b54cdabe485ec54517f3aa8b15
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(BUILD_SHARED_LIBS ON)

  FetchContent_Declare(
    rapids-cmake
    URL ${VELOX_rapids_cmake_SOURCE_URL}
    URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    rmm
    URL ${VELOX_rmm_SOURCE_URL}
    URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    kvikio
    URL ${VELOX_kvikio_SOURCE_URL}
    URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    cudf
    URL ${VELOX_cudf_SOURCE_URL}
    URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_MakeAvailable(cudf)

  # Apply cuDF patch for decimal stats filtering and verify it.
  set(VELOX_CUDF_PATCH_FILE
      "${CMAKE_CURRENT_LIST_DIR}/cudf/stats-filter-use-jit.patch")
  set(_cudf_predicate_file
      "${cudf_SOURCE_DIR}/cpp/src/io/parquet/predicate_pushdown.cpp")
  set(_cudf_stats_file
      "${cudf_SOURCE_DIR}/cpp/src/io/parquet/stats_filter_helpers.hpp")
  set(_cudf_predicate_marker "parquet stats filter: uses_fixed_point")
  set(_cudf_stats_marker "Decimal values are stored big-endian.")

  if (NOT EXISTS "${VELOX_CUDF_PATCH_FILE}")
    message(FATAL_ERROR "cuDF patch file not found: ${VELOX_CUDF_PATCH_FILE}")
  endif()
  if (NOT EXISTS "${_cudf_predicate_file}" OR NOT EXISTS "${_cudf_stats_file}")
    message(FATAL_ERROR
            "cuDF patch target files not found under ${cudf_SOURCE_DIR}")
  endif()

  file(READ "${_cudf_predicate_file}" _cudf_predicate_contents)
  file(READ "${_cudf_stats_file}" _cudf_stats_contents)
  string(FIND "${_cudf_predicate_contents}" "${_cudf_predicate_marker}"
         _cudf_predicate_marker_pos)
  string(FIND "${_cudf_stats_contents}" "${_cudf_stats_marker}"
         _cudf_stats_marker_pos)
  set(_cudf_patch_present TRUE)
  if (_cudf_predicate_marker_pos EQUAL -1)
    set(_cudf_patch_present FALSE)
  endif()
  if (_cudf_stats_marker_pos EQUAL -1)
    set(_cudf_patch_present FALSE)
  endif()

  if (NOT _cudf_patch_present)
    message(STATUS "Applying cuDF patch: ${VELOX_CUDF_PATCH_FILE}")
    execute_process(
      COMMAND patch -p1 --forward -i "${VELOX_CUDF_PATCH_FILE}"
      WORKING_DIRECTORY "${cudf_SOURCE_DIR}"
      RESULT_VARIABLE _cudf_patch_result
      OUTPUT_VARIABLE _cudf_patch_stdout
      ERROR_VARIABLE _cudf_patch_stderr)
    if (NOT _cudf_patch_result EQUAL 0)
      message(WARNING
              "cuDF patch command returned ${_cudf_patch_result}: "
              "${_cudf_patch_stderr}")
    endif()
  else()
    message(STATUS "cuDF patch already present; skipping apply.")
  endif()

  file(READ "${_cudf_predicate_file}" _cudf_predicate_contents_after)
  file(READ "${_cudf_stats_file}" _cudf_stats_contents_after)
  string(FIND "${_cudf_predicate_contents_after}" "${_cudf_predicate_marker}"
         _cudf_predicate_marker_pos_after)
  string(FIND "${_cudf_stats_contents_after}" "${_cudf_stats_marker}"
         _cudf_stats_marker_pos_after)
  if (_cudf_predicate_marker_pos_after EQUAL -1 OR
      _cudf_stats_marker_pos_after EQUAL -1)
    message(FATAL_ERROR "cuDF patch verification failed. Ensure patch is applied.")
  endif()
  unset(_cudf_predicate_contents)
  unset(_cudf_stats_contents)
  unset(_cudf_predicate_contents_after)
  unset(_cudf_stats_contents_after)
  unset(_cudf_predicate_marker_pos)
  unset(_cudf_stats_marker_pos)
  unset(_cudf_predicate_marker_pos_after)
  unset(_cudf_stats_marker_pos_after)
  unset(_cudf_patch_present)
  unset(_cudf_patch_result)
  unset(_cudf_patch_stdout)
  unset(_cudf_patch_stderr)
  unset(_cudf_predicate_file)
  unset(_cudf_stats_file)
  unset(_cudf_predicate_marker)
  unset(_cudf_stats_marker)

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
  unset(BUILD_TESTING CACHE)
endblock()
