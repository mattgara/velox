# GPU compression for UCX exchange

This optional path compresses cuDF exchange buffers on the sending GPU and
reconstructs them on the receiving GPU before `cudf::unpack`. It reduces
inter-worker traffic at the cost of GPU codec work.

Compression is disabled by default. Every worker in a query must support the
same wire format.

## Data path

The sender receives the normal `cudf::packed_columns` output. It keeps the
cuDF host metadata unchanged and divides the contiguous device buffer into
regions:

1. Fixed-width 32-bit and 64-bit numeric regions are candidates for
   frame-of-reference (FOR) or delta-FOR transforms.
2. The transformed byte planes are encoded with nvCOMP byte-ANS.
3. Validity buffers, offsets, strings, and other residual regions use
   byte-ANS when that reduces their size, or remain raw.
4. The complete candidate must be at least two percent smaller than the raw
   device buffer.
5. A compact region descriptor travels in the existing UCX metadata message.
6. The receiver reconstructs a byte-exact copy of the original device buffer
   and follows the unchanged `cudf::unpack` path.

FOR and delta-FOR are integer transforms. nvCOMP ANS is the entropy coder
applied to the resulting byte planes. nvCOMP is already supplied by the cuDF
build.

## Execution and transport

Encoding and decoding run on a bounded executor rather than the UCXX progress
thread. Broadcast destinations that share one packed buffer also share one
encode result.

Compression is skipped for the in-process same-worker path and for endpoints
that UCP reports as CUDA IPC. Known non-CUDA-IPC endpoints, including remote
TCP or RDMA endpoints, are eligible. If transport discovery fails, the sender
leaves the payload raw.

## Configuration

Set these properties in the native worker configuration:

| Property | Default | Meaning |
| --- | --- | --- |
| `cudf.exchange_compression` | `none` | `none` sends raw data. `column` always tries compression. `column-adaptive` uses measured profitability. |
| `cudf.exchange_compression_pipeline_threads` | `1` | Maximum concurrent codec tasks per worker. Accepted values are 1 through 4. |
| `cudf.exchange_compression_min_bytes` | `0` | Leave chunks smaller than this many bytes raw. |
| `cudf.exchange_compression_safety_margin` | `1.10` | Required ratio of estimated transfer savings to measured codec cost. |

For example:

```properties
cudf.exchange=true
cudf.exchange_compression=column-adaptive
cudf.exchange_compression_pipeline_threads=1
cudf.exchange_compression_min_bytes=16777216
cudf.exchange_compression_safety_margin=1.10
```

Setting `cudf.exchange_compression=none` restores the raw UCX data path.
Adaptive mode maintains a small online model per query stage. Initial chunks
run the normal codec and provide encode, real UCX transfer, and
compression-ratio samples. Candidates that are sent compressed also provide
decode samples. Later chunks are compressed only when estimated transfer
savings exceed measured codec cost by the safety margin. A stage that selects
raw transfer is periodically reprobed. Probes are sent chunks, not duplicate
encode or local decode passes.

## Build and validation

Build Velox with cuDF, CUDA, and UCX exchange enabled. The minimum relevant
CMake options are:

```bash
cmake -S . -B _build/release \
  -DVELOX_ENABLE_CUDF=ON \
  -DVELOX_ENABLE_UCX_EXCHANGE=ON \
  -DVELOX_BUILD_TESTING=ON
```

The transport-neutral codec lives in
`velox/experimental/cudf/compression` and links nvCOMP through the dependency
already supplied by the cuDF build.

With `VELOX_BUILD_TESTING=ON`, build the focused test targets:

```bash
cmake --build _build/release -j4 --target \
  cudf_compression_test \
  ucx_exchange_test
```

The cost-model unit cases are part of `ucx_exchange_test`. Run only those
cases with:

```bash
_build/release/velox/experimental/ucx-exchange/tests/ucx_exchange_test \
  --gtest_filter='UcxCompressionCostModelTest.*'
```

The standalone compression target contains descriptor and GPU round-trip
coverage. It requires a CUDA-capable test environment and does not link the
UCX exchange target:

```bash
_build/release/velox/experimental/cudf/compression/tests/cudf_compression_test
```

An end-to-end deployment must also confirm that all workers use the same
build, UCX rather than HTTP carries the exchange, and compressed and raw runs
return identical results.

## Implementation map

The reusable API and ownership contract are documented in
`../cudf/compression/README.md`.

- `cudf/compression/PackedColumnsCodec.*`: packed-layout inspection, integer
  transforms, opaque descriptor serialization, and reconstruction.
- `cudf/compression/detail/AnsCodec.*`: private nvCOMP ANS implementation.
- `UcxCompressionCostModel.*`: stage-level runtime profitability selection.
- `Communicator.*`: bounded codec executor and lifecycle.
- `UcxExchangeServer.*`: transport gate, encode, broadcast reuse, and send.
- `UcxExchangeSource.*`: descriptor parsing, decode, and enqueue.
- `EndpointRef.*`: UCP transport discovery.
