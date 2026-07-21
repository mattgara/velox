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
#include "velox/experimental/ucx-exchange/UcxCompression.h"

#include <mutex>

#include <cuda_runtime.h>

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/dietgpu/ans/GpuANSCodec.h"
#include "velox/experimental/ucx-exchange/dietgpu/utils/StackDeviceMemory.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr int kProbBits = 10;

#define UCX_CUDA_CHECK(expr)                          \
  do {                                                \
    cudaError_t err = (expr);                         \
    VELOX_CHECK_EQ(                                   \
        err,                                          \
        cudaSuccess,                                  \
        "CUDA error in ucx-exchange compression: {}", \
        cudaGetErrorString(err));                     \
  } while (0)

// DietGPU scratch memory, sized so encode/decode of a full chunk batch does
// not fall back to cudaMalloc (see StackDeviceMemory warnings).
dietgpu::StackDeviceMemory& stackMemory() {
  static std::once_flag flag;
  static std::unique_ptr<dietgpu::StackDeviceMemory> mem;
  std::call_once(flag, [] {
    int device = 0;
    UCX_CUDA_CHECK(cudaGetDevice(&device));
    mem = std::make_unique<dietgpu::StackDeviceMemory>(
        device, 768u << 20 /* 768 MiB scratch */);
  });
  return *mem;
}

std::vector<std::pair<const uint8_t*, uint32_t>> segments(
    const void* src,
    std::size_t size) {
  std::vector<std::pair<const uint8_t*, uint32_t>> segs;
  const auto* base = static_cast<const uint8_t*>(src);
  for (std::size_t off = 0; off < size; off += kCompressSegmentBytes) {
    segs.emplace_back(
        base + off,
        static_cast<uint32_t>(std::min(kCompressSegmentBytes, size - off)));
  }
  return segs;
}

} // namespace

CompressResult compressBlob(
    const void* src,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain,
    std::size_t minBytes) {
  CompressResult result;
  if (size < minBytes) {
    return result;
  }
  const auto segs = segments(src, size);
  const uint32_t numSegs = segs.size();

  // Strided scratch output: maxCompressedSize per segment.
  const uint32_t maxCompSeg =
      dietgpu::getMaxCompressedSize(kCompressSegmentBytes);
  rmm::device_buffer scratch(
      static_cast<std::size_t>(numSegs) * maxCompSeg, stream);
  rmm::device_buffer outSizesDev(numSegs * sizeof(uint32_t), stream);

  std::vector<const void*> inPtrs(numSegs);
  std::vector<uint32_t> inSizes(numSegs);
  std::vector<void*> outPtrs(numSegs);
  for (uint32_t i = 0; i < numSegs; ++i) {
    inPtrs[i] = segs[i].first;
    inSizes[i] = segs[i].second;
    outPtrs[i] = static_cast<uint8_t*>(scratch.data()) +
        static_cast<std::size_t>(i) * maxCompSeg;
  }

  dietgpu::ansEncodeBatchPointer(
      stackMemory(),
      dietgpu::ANSCodecConfig(kProbBits, /*useChecksum=*/true),
      numSegs,
      inPtrs.data(),
      inSizes.data(),
      /*histogram_dev=*/nullptr,
      outPtrs.data(),
      static_cast<uint32_t*>(outSizesDev.data()),
      stream.value());

  // Read back per-segment compressed sizes (one sync; the send path syncs
  // before UCX hand-off anyway).
  result.segSizes.resize(numSegs);
  UCX_CUDA_CHECK(cudaMemcpyAsync(
      result.segSizes.data(),
      outSizesDev.data(),
      numSegs * sizeof(uint32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  UCX_CUDA_CHECK(cudaStreamSynchronize(stream.value()));

  std::size_t total = 0;
  for (auto s : result.segSizes) {
    total += s;
  }
  if (total == 0 ||
      static_cast<double>(total) > (1.0 - minGain) * size) {
    return result; // did not pay; send uncompressed
  }

  // Compact the strided segments into one contiguous buffer.
  result.data = rmm::device_buffer(total, stream);
  std::size_t off = 0;
  for (uint32_t i = 0; i < numSegs; ++i) {
    UCX_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(result.data.data()) + off,
        outPtrs[i],
        result.segSizes[i],
        cudaMemcpyDeviceToDevice,
        stream.value()));
    off += result.segSizes[i];
  }
  result.used = true;
  return result;
}

rmm::device_buffer decompressBlob(
    const void* src,
    const std::vector<uint32_t>& segSizes,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream) {
  const uint32_t numSegs = segSizes.size();
  VELOX_CHECK_GT(numSegs, 0, "decompressBlob: empty segment list");
  rmm::device_buffer out(uncompressedBytes, stream);

  std::vector<const void*> inPtrs(numSegs);
  std::vector<void*> outPtrs(numSegs);
  std::vector<uint32_t> outCaps(numSegs);
  std::size_t inOff = 0;
  std::size_t outOff = 0;
  for (uint32_t i = 0; i < numSegs; ++i) {
    inPtrs[i] = static_cast<const uint8_t*>(src) + inOff;
    inOff += segSizes[i];
    outPtrs[i] = static_cast<uint8_t*>(out.data()) + outOff;
    const auto cap = static_cast<uint32_t>(
        std::min(kCompressSegmentBytes, uncompressedBytes - outOff));
    outCaps[i] = cap;
    outOff += cap;
  }
  VELOX_CHECK_EQ(
      outOff, uncompressedBytes, "decompressBlob: segment/output mismatch");

  auto status = dietgpu::ansDecodeBatchPointer(
      stackMemory(),
      dietgpu::ANSCodecConfig(kProbBits, /*useChecksum=*/true),
      numSegs,
      inPtrs.data(),
      outPtrs.data(),
      outCaps.data(),
      /*outSuccess_dev=*/nullptr,
      /*outSize_dev=*/nullptr,
      stream.value());
  VELOX_CHECK(
      status.error == dietgpu::ANSDecodeError::None,
      "ucx-exchange rANS decode failed (checksum or corrupt frame)");
  return out;
}

} // namespace facebook::velox::ucx_exchange
