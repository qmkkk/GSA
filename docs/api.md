# Query semantics

## Coordinate and execution contract

Inputs are contiguous CUDA FP32 tensors of shape `(N, 3)` containing finite coordinates. Each session is used sequentially on the same device and CUDA stream on which it was created. The native allocator serves one device per process. Inputs and outputs have separate storage. Keep tensors alive until queued CUDA operations complete; close the session on its original stream. `torch.inference_mode()` is recommended.

The native build fixes the distance operation order to `((dx*dx + dy*dy) + dz*dz)` in FP32 with contraction disabled and subnormal values retained. BVH lower bounds use downward-rounded operations. Finite coordinates may still produce overflowing squared distances: 3NN skips non-finite distances and fills remaining entries with `inf` and ID `0`.

## Radius query

`session.ball_into(q, x, radius, k, out)` writes an int32 `(len(q), k)` tensor. Support `x` contains at least one point; `1 <= k <= 32`; radius is finite, nonnegative and representable as FP32. Qualification is strict: `distance_squared < radius_squared`.

Qualifying points are selected in increasing **original support ID** order. If fewer than `k` qualify, the first qualifying ID pads the row. If none qualify, the row is zero. A zero radius therefore produces zero rows.

The cache owns up to 32 results per query. A cache hit requires equal values and order for the entire support tensor, equal radius, and equal values for the requested query prefix. Output writes cannot modify the private cached result. Shortening the support changes the query problem and does not admit a radius-result cache hit by itself.

## Three-nearest-neighbor query

`session.nn_into(q, x, distances, ids)` writes FP32 squared distances and int32 IDs, both `(len(q), 3)`. Support has at least three points. Ties are resolved by increasing original ID. Empty query batches are valid for both operations.

## Index reuse

For a root with `N` points, the prefix index stores bounds for `N`, `floor(N/4)`, `floor(N/16)`, and `floor(N/64)`. QuerySession verifies support values against its owned root snapshot before selecting a reusable index. An unrelated support gets its own index. A session should span one meaningful forward or short query chain so its retained coordinates and cached rows have a clear lifetime.

Supports of at most `threshold` points use the original-order scan fallback (`threshold=1024` by default). This threshold is part of the measured implementation.

## Statistics

`session.stats` reports builds, index reuse, query calls, radius cache hits, coordinate comparisons, snapshot bytes, and cache bytes. These byte counts describe retained allocations, not total device memory. `close(collect=True)` additionally collects traversal-stack fallback counts. A traversal-stack overflow falls back to an exact full scan.

`gsa.runtime.Tree` is the lower-level index interface used by the fixed-graph adapter. Its coordinate tensor is borrowed and must remain immutable until the tree closes. QuerySession is the default interface for callers that mutate their inputs between calls.
