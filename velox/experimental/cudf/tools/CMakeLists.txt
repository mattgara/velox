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

# Build replay utility even when VELOX_BUILD_TESTING=OFF.
add_executable(velox_cudf_hashagg_replay ../tests/HashAggReplay.cpp)
target_link_libraries(velox_cudf_hashagg_replay velox_cudf_exec)

add_executable(velox_cudf_decimal_groupby_repro DecimalGroupbyRepro.cpp)
target_link_libraries(velox_cudf_decimal_groupby_repro velox_cudf_exec)
