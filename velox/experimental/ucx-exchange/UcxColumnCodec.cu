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
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"
#include "velox/experimental/ucx-exchange/UcxCompression.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "velox/experimental/ucx-exchange/dietgpu/ans/GpuANSCodec.h"
#include "velox/experimental/ucx-exchange/dietgpu/utils/StackDeviceMemory.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr int kProbBits = 10;
// Racecar default: correctness is gated end-to-end by result validation;
// frame checksums (two extra full passes) are a debug knob only.
constexpr bool kUseChecksum = false;
constexpr std::size_t kMinTypedElems = 4096;

// DietGPU kernels use word loads; every device pointer handed to them must
// be 16-byte aligned. Planes use an aligned stride; wire segments are placed
// at 16-byte boundaries (true sizes travel in descriptors, walkers round).
inline std::size_t roundUp16(std::size_t v) {
  return (v + 15) & ~static_cast<std::size_t>(15);
}
inline uint32_t alignedStride(uint32_t n) {
  return (n + 15u) & ~15u;
}
constexpr std::size_t kMinResidualBytes = 1u << 16;

#define UCX_CUDA_CHECK(expr)                                       \
  do {                                                             \
    cudaError_t err = (expr);                                      \
    if (err != cudaSuccess) {                                      \
      throw std::runtime_error(fmt::format(                        \
          "CUDA error in ucx-exchange column codec: {}",           \
          cudaGetErrorString(err)));                               \
    }                                                              \
  } while (0)

// Per-call DietGPU scratch arena backed by rmm (stream-ordered): safe under
// concurrent encode/decode on different streams, no shared state.
constexpr std::size_t kPlaneArenaBytes = 256u << 20;

struct PlaneArena {
  rmm::device_buffer buffer;
  dietgpu::StackDeviceMemory stack;
  explicit PlaneArena(rmm::cuda_stream_view stream)
      : buffer(kPlaneArenaBytes, stream),
        stack(
            [] {
              int device = 0;
              cudaGetDevice(&device);
              return device;
            }(),
            buffer.data(),
            kPlaneArenaBytes) {}
};

// Subtracts base and splits into w byte planes (SoA, plane-major).
template <typename T>
__global__ void subSplitKernel(
    const T* values,
    int64_t base,
    uint8_t* planes,
    uint32_t n,
    uint32_t stride,
    int width) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t adjusted = static_cast<uint64_t>(
      static_cast<int64_t>(values[i]) - base);
  for (int k = 0; k < width; ++k) {
    planes[static_cast<uint64_t>(k) * stride + i] =
        static_cast<uint8_t>((adjusted >> (8 * k)) & 0xff);
  }
}

// Merges w byte planes and adds base.
template <typename T>
__global__ void recombAddKernel(
    const uint8_t* planes,
    int64_t base,
    T* out,
    uint32_t n,
    uint32_t stride,
    int width) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t adjusted = 0;
  for (int k = 0; k < width; ++k) {
    adjusted |=
        static_cast<uint64_t>(planes[static_cast<uint64_t>(k) * stride + i])
        << (8 * k);
  }
  out[i] = static_cast<T>(static_cast<int64_t>(adjusted) + base);
}

// zigzag(v[i] - v[i-1]); element 0 gets 0 (first value travels in the
// descriptor).
template <typename T>
__global__ void zigzagDeltaKernel(const T* values, int64_t* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int64_t delta = (i == 0)
      ? 0
      : static_cast<int64_t>(values[i]) - static_cast<int64_t>(values[i - 1]);
  out[i] = (delta << 1) ^ (delta >> 63);
}

// Un-zigzags in place (int64 zigzag values -> signed deltas).
__global__ void unZigzagKernel(int64_t* values, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  uint64_t z = static_cast<uint64_t>(values[i]);
  values[i] = static_cast<int64_t>((z >> 1) ^ (~(z & 1) + 1));
}

// Adds `first` to every prefix-summed delta and narrows to T.
template <typename T>
__global__ void finalizeDeltaKernel(
    const int64_t* summed,
    int64_t first,
    T* out,
    uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = static_cast<T>(summed[i] + first);
}

int planesForRange(uint64_t range) {
  int width = 1;
  while ((range >> (8 * width)) != 0 && width < 8) {
    ++width;
  }
  return width;
}

struct TypedRegion {
  std::size_t offset; // byte offset in blob
  std::size_t elems;
  int width; // element width, 4 or 8
};

// Recursively collects fixed-width numeric data-buffer regions.
void collectTypedRegions(
    const cudf::column_view& col,
    const uint8_t* blobBase,
    std::vector<TypedRegion>& out) {
  const auto type = col.type();
  if (col.size() >= static_cast<cudf::size_type>(kMinTypedElems) &&
      cudf::is_fixed_width(type) && col.head<uint8_t>() != nullptr) {
    const int width = cudf::size_of(type);
    if (width == 4 || width == 8) {
      const uint8_t* data =
          col.head<uint8_t>() + static_cast<std::size_t>(col.offset()) * width;
      out.push_back(TypedRegion{
          static_cast<std::size_t>(data - blobBase),
          static_cast<std::size_t>(col.size()),
          width});
    }
  }
  for (auto childIndex = 0; childIndex < col.num_children(); ++childIndex) {
    collectTypedRegions(col.child(childIndex), blobBase, out);
  }
}

// Encodes `w` byte planes of n bytes each (plane-major in `planes`) with one
// strided rANS batch; returns compacted output + per-plane sizes.
std::pair<rmm::device_buffer, std::vector<int64_t>> encodePlanes(
    const uint8_t* planes,
    uint32_t n,
    uint32_t stride,
    int width,
    rmm::cuda_stream_view stream) {
  const uint32_t maxComp =
      alignedStride(dietgpu::getMaxCompressedSize(n));
  PlaneArena arena(stream);
  rmm::device_buffer scratch(static_cast<std::size_t>(width) * maxComp, stream);
  rmm::device_buffer sizesDev(width * sizeof(uint32_t), stream);
  dietgpu::ansEncodeBatchStride(
      arena.stack,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      width,
      planes,
      n,
      stride,
      nullptr,
      scratch.data(),
      maxComp,
      static_cast<uint32_t*>(sizesDev.data()),
      stream.value());
  std::vector<uint32_t> sizes(width);
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      sizes.data(),
      sizesDev.data(),
      width * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));

  std::size_t total = 0;
  for (auto s : sizes) {
    total += roundUp16(s);
  }
  rmm::device_buffer out(total, stream);
  std::size_t off = 0;
  for (int k = 0; k < width; ++k) {
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(out.data()) + off,
        static_cast<const uint8_t*>(scratch.data()) +
            static_cast<std::size_t>(k) * maxComp,
        sizes[k],
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += roundUp16(sizes[k]);
  }
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return {std::move(out), std::vector<int64_t>(sizes.begin(), sizes.end())};
}

// Decodes `segSizes.size()` planes of n bytes each into plane-major scratch.
rmm::device_buffer decodePlanes(
    const uint8_t* src,
    const std::vector<int64_t>& segSizes,
    uint32_t n,
    rmm::cuda_stream_view stream) {
  const auto width = segSizes.size();
  const uint32_t stride = alignedStride(n);
  rmm::device_buffer planes(static_cast<std::size_t>(width) * stride, stream);
  PlaneArena arena(stream);
  std::vector<const void*> inPtrs(width);
  std::vector<void*> outPtrs(width);
  std::vector<uint32_t> outCaps(width, n);
  std::size_t off = 0;
  for (std::size_t k = 0; k < width; ++k) {
    inPtrs[k] = src + off;
    off += roundUp16(segSizes[k]);
    outPtrs[k] = static_cast<uint8_t*>(planes.data()) + k * stride;
  }
  auto status = dietgpu::ansDecodeBatchPointer(
      arena.stack,
      dietgpu::ANSCodecConfig(kProbBits, kUseChecksum),
      width,
      inPtrs.data(),
      outPtrs.data(),
      outCaps.data(),
      nullptr,
      nullptr,
      stream.value());
  if (status.error != dietgpu::ANSDecodeError::None) {
    throw std::runtime_error("ucx-exchange column codec: plane decode failed");
  }
  // Settle before the arena (stream-ordered) frees and callers consume planes.
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  return planes;
}

template <typename T>
void encodeTypedRegion(
    const uint8_t* blobBase,
    const TypedRegion& region,
    rmm::cuda_stream_view stream,
    std::vector<EncodedRegion>& regions,
    std::vector<rmm::device_buffer>& payloads) {
  const T* values = reinterpret_cast<const T*>(blobBase + region.offset);
  const uint32_t n = region.elems;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // Value min/max.
  auto valuesPtr = thrust::device_pointer_cast(values);
  auto valueMinMax = thrust::minmax_element(
      thrust::cuda::par.on(stream.value()), valuesPtr, valuesPtr + n);
  T hostMin;
  T hostMax;
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      &hostMin, valueMinMax.first.get(), sizeof(T),
      cudaMemcpyDeviceToHost, stream.value()));
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      &hostMax, valueMinMax.second.get(), sizeof(T),
      cudaMemcpyDeviceToHost, stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  const int64_t base = static_cast<int64_t>(hostMin);
  const uint64_t forRange = static_cast<uint64_t>(
      static_cast<int64_t>(hostMax) - base);
  const int forWidth = planesForRange(forRange);

  // Zigzag deltas + their max (min is >= 0 by construction).
  rmm::device_buffer deltas(static_cast<std::size_t>(n) * 8, stream);
  auto* deltasPtr = static_cast<int64_t*>(deltas.data());
  zigzagDeltaKernel<T>
      <<<blocks, threads, 0, stream.value()>>>(values, deltasPtr, n);
  auto deltaThrust = thrust::device_pointer_cast(deltasPtr);
  auto deltaMaxIt = thrust::max_element(
      thrust::cuda::par.on(stream.value()), deltaThrust, deltaThrust + n);
  int64_t deltaMax;
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      &deltaMax, deltaMaxIt.get(), sizeof(int64_t),
      cudaMemcpyDeviceToHost, stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  const int deltaWidth = planesForRange(static_cast<uint64_t>(deltaMax));

  EncodedRegion out;
  out.blobOffset = region.offset;
  out.rawBytes = static_cast<int64_t>(region.elems) * region.width;
  out.elemWidth = region.width;

  const uint32_t stride = alignedStride(n);
  rmm::device_buffer planes(
      static_cast<std::size_t>(std::min(forWidth, deltaWidth)) * stride,
      stream);
  if (deltaWidth < forWidth) {
    out.codec = RegionCodec::kDeltaFor;
    out.base = 0;
    T firstValue;
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        &firstValue, values, sizeof(T), cudaMemcpyDeviceToHost,
        stream.value()));
    UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
    out.first = static_cast<int64_t>(firstValue);
    subSplitKernel<int64_t><<<blocks, threads, 0, stream.value()>>>(
        deltasPtr,
        0,
        static_cast<uint8_t*>(planes.data()),
        n,
        stride,
        deltaWidth);
    auto [payload, sizes] = encodePlanes(
        static_cast<const uint8_t*>(planes.data()),
        n,
        stride,
        deltaWidth,
        stream);
    out.segSizes = std::move(sizes);
    payloads.push_back(std::move(payload));
  } else {
    out.codec = RegionCodec::kFor;
    out.base = base;
    subSplitKernel<T><<<blocks, threads, 0, stream.value()>>>(
        values,
        base,
        static_cast<uint8_t*>(planes.data()),
        n,
        stride,
        forWidth);
    auto [payload, sizes] = encodePlanes(
        static_cast<const uint8_t*>(planes.data()),
        n,
        stride,
        forWidth,
        stream);
    out.segSizes = std::move(sizes);
    payloads.push_back(std::move(payload));
  }
  regions.push_back(std::move(out));
}

template <typename T>
void decodeTypedRegion(
    const uint8_t* src,
    const EncodedRegion& region,
    uint8_t* blobBase,
    rmm::cuda_stream_view stream) {
  const uint32_t n = region.rawBytes / region.elemWidth;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  auto planes = decodePlanes(src, region.segSizes, n, stream);
  const uint32_t stride = alignedStride(n);
  T* out = reinterpret_cast<T*>(blobBase + region.blobOffset);

  if (region.codec == RegionCodec::kFor) {
    recombAddKernel<T><<<blocks, threads, 0, stream.value()>>>(
        static_cast<const uint8_t*>(planes.data()),
        region.base,
        out,
        n,
        stride,
        region.segSizes.size());
    return;
  }
  // kDeltaFor: planes -> zigzag deltas -> signed deltas -> prefix sum -> +first.
  rmm::device_buffer deltas(static_cast<std::size_t>(n) * 8, stream);
  auto* deltasPtr = static_cast<int64_t*>(deltas.data());
  recombAddKernel<int64_t><<<blocks, threads, 0, stream.value()>>>(
      static_cast<const uint8_t*>(planes.data()),
      0,
      deltasPtr,
      n,
      stride,
      region.segSizes.size());
  unZigzagKernel<<<blocks, threads, 0, stream.value()>>>(deltasPtr, n);
  std::size_t tempBytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, tempBytes, deltasPtr, deltasPtr, n, stream.value());
  rmm::device_buffer temp(tempBytes, stream);
  cub::DeviceScan::InclusiveSum(
      temp.data(), tempBytes, deltasPtr, deltasPtr, n, stream.value());
  finalizeDeltaKernel<T><<<blocks, threads, 0, stream.value()>>>(
      deltasPtr, region.first, out, n);
}

} // namespace

PackedCompressResult compressPacked(
    const uint8_t* metadata,
    const void* gpuData,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain) {
  PackedCompressResult result;
  const auto* blobBase = static_cast<const uint8_t*>(gpuData);

  std::vector<TypedRegion> typed;
  auto view = cudf::unpack(metadata, blobBase);
  for (const auto& col : view) {
    collectTypedRegions(col, blobBase, typed);
  }
  std::sort(typed.begin(), typed.end(), [](const auto& a, const auto& b) {
    return a.offset < b.offset;
  });

  std::vector<EncodedRegion> regions;
  std::vector<rmm::device_buffer> payloads;
  std::size_t cursor = 0;

  auto addResidual = [&](std::size_t offset, std::size_t bytes) {
    if (bytes == 0) {
      return;
    }
    EncodedRegion region;
    region.blobOffset = offset;
    region.rawBytes = bytes;
    if (bytes >= kMinResidualBytes) {
      // Stage into a fresh (aligned) buffer: gap offsets inside the blob are
      // arbitrary and dietgpu requires aligned input pointers.
      rmm::device_buffer staged(bytes, stream);
      UCX_CUDA_CHECK(cudaMemcpyAsync(
          staged.data(),
          blobBase + offset,
          bytes,
          cudaMemcpyDeviceToDevice,
          stream.value()));
      auto compressed = compressBlob(staged.data(), bytes, stream, minGain, 1);
      if (compressed.used) {
        region.codec = RegionCodec::kByteRans;
        region.segSizes.assign(
            compressed.segSizes.begin(), compressed.segSizes.end());
        payloads.push_back(std::move(compressed.data));
        regions.push_back(std::move(region));
        return;
      }
    }
    region.codec = RegionCodec::kRaw;
    payloads.emplace_back(); // raw regions copy from the original blob
    regions.push_back(std::move(region));
  };

  for (const auto& region : typed) {
    const std::size_t regionBytes = region.elems * region.width;
    if (region.offset < cursor) {
      continue; // overlap safety; leave to residual coverage of earlier pass
    }
    addResidual(cursor, region.offset - cursor);
    if (region.width == 8) {
      encodeTypedRegion<int64_t>(blobBase, region, stream, regions, payloads);
    } else {
      encodeTypedRegion<int32_t>(blobBase, region, stream, regions, payloads);
    }
    cursor = region.offset + regionBytes;
  }
  addResidual(cursor, size - cursor);

  std::size_t total = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    total += roundUp16(
        regions[i].codec == RegionCodec::kRaw
            ? static_cast<std::size_t>(regions[i].rawBytes)
            : payloads[i].size());
  }
  if (static_cast<double>(total) > (1.0 - minGain) * size) {
    return result;
  }

  result.data = rmm::device_buffer(total, stream);
  std::size_t off = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const bool raw = regions[i].codec == RegionCodec::kRaw;
    const auto bytes = raw ? static_cast<std::size_t>(regions[i].rawBytes)
                           : payloads[i].size();
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + off,
        raw ? blobBase + regions[i].blobOffset
            : static_cast<const uint8_t*>(payloads[i].data()),
        bytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += roundUp16(bytes);
  }
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
  result.regions = std::move(regions);
  result.used = true;
  return result;
}

rmm::device_buffer decompressPacked(
    const void* src,
    const std::vector<EncodedRegion>& regions,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer blob(uncompressedBytes, stream);
  auto* blobBase = static_cast<uint8_t*>(blob.data());
  const auto* wire = static_cast<const uint8_t*>(src);
  std::size_t off = 0;

  for (const auto& region : regions) {
    std::size_t encodedBytes = 0; // true wire footprint before padding
    switch (region.codec) {
      case RegionCodec::kRaw:
        encodedBytes = region.rawBytes;
        UCX_CUDA_CHECK(cudaMemcpyAsync(
            blobBase + region.blobOffset,
            wire + off,
            encodedBytes,
            cudaMemcpyDeviceToDevice,
            stream.value()));
        break;
      case RegionCodec::kByteRans: {
        for (auto s : region.segSizes) {
          encodedBytes += roundUp16(s);
        }
        std::vector<uint32_t> segSizes(
            region.segSizes.begin(), region.segSizes.end());
        auto decoded =
            decompressBlob(wire + off, segSizes, region.rawBytes, stream);
        UCX_CUDA_CHECK(cudaMemcpyAsync(
            blobBase + region.blobOffset,
            decoded.data(),
            region.rawBytes,
            cudaMemcpyDeviceToDevice,
            stream.value()));
        UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));
        break;
      }
      case RegionCodec::kFor:
      case RegionCodec::kDeltaFor: {
        for (auto s : region.segSizes) {
          encodedBytes += roundUp16(s);
        }
        if (region.elemWidth == 8) {
          decodeTypedRegion<int64_t>(wire + off, region, blobBase, stream);
        } else {
          decodeTypedRegion<int32_t>(wire + off, region, blobBase, stream);
        }
        break;
      }
    }
    off += roundUp16(encodedBytes);
  }
  return blob;
}

void serializeRegions(
    const PackedCompressResult& result,
    std::size_t uncompressedBytes,
    std::vector<int64_t>& out) {
  out.clear();
  out.push_back(kPerColumnMagic);
  out.push_back(static_cast<int64_t>(uncompressedBytes));
  out.push_back(static_cast<int64_t>(result.regions.size()));
  for (const auto& region : result.regions) {
    out.push_back(region.blobOffset);
    out.push_back(region.rawBytes);
    out.push_back(static_cast<int64_t>(region.codec));
    out.push_back(region.elemWidth);
    out.push_back(region.base);
    out.push_back(region.first);
    out.push_back(static_cast<int64_t>(region.segSizes.size()));
    for (auto s : region.segSizes) {
      out.push_back(s);
    }
  }
}

bool deserializeRegions(
    const std::vector<int64_t>& in,
    std::vector<EncodedRegion>& regions,
    std::size_t& uncompressedBytes) {
  if (in.size() < 3 || in[0] != kPerColumnMagic) {
    return false;
  }
  uncompressedBytes = static_cast<std::size_t>(in[1]);
  const auto numRegions = static_cast<std::size_t>(in[2]);
  std::size_t pos = 3;
  regions.clear();
  regions.reserve(numRegions);
  for (std::size_t r = 0; r < numRegions; ++r) {
    if (pos + 7 > in.size()) {
      return false;
    }
    EncodedRegion region;
    region.blobOffset = in[pos++];
    region.rawBytes = in[pos++];
    region.codec = static_cast<RegionCodec>(in[pos++]);
    region.elemWidth = static_cast<int32_t>(in[pos++]);
    region.base = in[pos++];
    region.first = in[pos++];
    const auto numSegs = static_cast<std::size_t>(in[pos++]);
    if (pos + numSegs > in.size()) {
      return false;
    }
    region.segSizes.assign(in.begin() + pos, in.begin() + pos + numSegs);
    pos += numSegs;
    regions.push_back(std::move(region));
  }
  return pos == in.size();
}

} // namespace facebook::velox::ucx_exchange
