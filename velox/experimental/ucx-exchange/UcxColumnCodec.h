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
#pragma once

#include <cstdint>
#include <vector>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

/// Hook v2: per-column JIT encoding of the packed exchange blob.
///
/// The sender walks the cudf::unpack'ed table view (host-side pointer math)
/// and encodes each fixed-width numeric column's data buffer with a
/// frame-of-reference transform (plain FOR, or zigzag-delta + FOR for
/// sequential data), byte-plane split, and rANS over the value-bearing
/// planes. Regions not covered by a typed encoder (validity, offsets,
/// strings) fall back to whole-region byte-rANS or raw copy. The receiver
/// reconstructs the packed blob byte-exactly, so cudf::unpack downstream is
/// unchanged.
namespace facebook::velox::ucx_exchange {

/// Region codecs carried in the wire descriptor.
enum class RegionCodec : int64_t {
  kRaw = 0,
  kByteRans = 1,
  kFor = 2, // subtract-min + byte planes + rANS
  kDeltaFor = 3, // zigzag delta, then kFor on the deltas
};

/// One encoded region of the packed blob.
/// Wire layout in MetadataMsg::remainingBytes (all int64):
///   [kPerColumnMagic, uncompressedBlobBytes, numRegions,
///    per region: blobOffset, rawBytes, codec, elemWidth, base, first,
///                numSegs, segSize0..segSizeN-1]
struct EncodedRegion {
  int64_t blobOffset{0};
  int64_t rawBytes{0};
  RegionCodec codec{RegionCodec::kRaw};
  int32_t elemWidth{1}; // element byte width for typed codecs (4 or 8)
  int64_t base{0}; // FOR base (minimum) for kFor / delta-min for kDeltaFor
  int64_t first{0}; // first element value for kDeltaFor
  std::vector<int64_t> segSizes; // rANS plane/segment byte sizes
};

/// Codec id for remainingBytes[0] distinguishing v2 from v1 (kByteRans=1).
constexpr int64_t kPerColumnMagic = 2;

struct PackedCompressResult {
  bool used{false};
  rmm::device_buffer data; // concatenated encoded regions
  std::vector<EncodedRegion> regions;
};

/// Compresses the packed blob per column. `metadata`/`gpuData` are the
/// cudf::pack outputs (host metadata, device blob of `size` bytes).
/// Returns used=false when the encoded total does not undercut `size` by
/// at least minGain.
PackedCompressResult compressPacked(
    const uint8_t* metadata,
    const void* gpuData,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain = 0.02);

/// Reconstructs the packed blob (byte-exact) from encoded regions.
rmm::device_buffer decompressPacked(
    const void* src,
    const std::vector<EncodedRegion>& regions,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream);

/// Serialization to/from the MetadataMsg remainingBytes vector.
void serializeRegions(
    const PackedCompressResult& result,
    std::size_t uncompressedBytes,
    std::vector<int64_t>& out);
bool deserializeRegions(
    const std::vector<int64_t>& in,
    std::vector<EncodedRegion>& regions,
    std::size_t& uncompressedBytes);

} // namespace facebook::velox::ucx_exchange
