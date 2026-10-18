# SPDX-License-Identifier: Apache-2.0
"""Exact coordinate-value reuse across ordered point-query call sequences.

The session retains its own support/query snapshots and private result tensors.
Callers may change their tensors between synchronous calls. Reuse checks compare
current coordinates with the retained values before selecting an existing plan.
"""

from . import runtime as er
from . import reference as rt
import torch
import math


class QuerySession:
    def __init__(self, ball_reference=None, nn_reference=None, threshold=1024):
        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError("threshold must be a nonnegative integer")
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.ball_reference = ball_reference or rt.ball_query_wrapper
        self.nn_reference = nn_reference or rt.three_nn_wrapper
        self.threshold = threshold
        self.trees = []
        self.memo = []
        self.closed = False
        self.stats = {
            "builds": 0,
            "tree_reuses": 0,
            "ball_calls": 0,
            "ball_hits": 0,
            "nn_calls": 0,
            "coordinate_comparisons": 0,
            "snapshot_bytes": 0,
            "cache_bytes": 0,
            "fallbacks": 0,
        }

    def _equal(self, a, b):
        if a.shape != b.shape or a.device != b.device or a.dtype != b.dtype:
            return False
        self.stats["coordinate_comparisons"] += 1
        return torch.equal(a, b)

    def _snapshot(self, x):
        y = x.clone()
        self.stats["snapshot_bytes"] += y.numel() * y.element_size()
        return y

    def _check(self, x):
        if not (
            not self.closed
            and x.ndim == 2
            and x.shape[1] == 3
            and x.is_contiguous()
            and x.dtype == torch.float32
            and x.is_cuda
            and x.device.index == self.device == torch.cuda.current_device()
            and torch.cuda.current_stream().cuda_stream == self.stream
        ):
            raise ValueError(
                "Use an open session with contiguous FP32 (N,3) tensors on its original CUDA device and stream"
            )

    def _tree(self, x):
        self._check(x)
        m = len(x)
        for root, tree, sizes in self.trees:
            if m in sizes and self._equal(x, root[:m]):
                self.stats["tree_reuses"] += 1
                return tree
        root = self._snapshot(x)
        sizes = []
        n = len(root)
        for _ in range(4):
            sizes.append(n)
            n //= 4
        tree = er.Tree(root, prefix=True)
        self.trees.append((root, tree, sizes))
        self.stats["builds"] += 1
        return tree

    def ball_into(self, q, x, r, k, out):
        self._check(x)
        self._check(q)
        if not (
            len(x) >= 1
            and isinstance(k, int)
            and 1 <= k <= 32
            and out.shape == (len(q), k)
            and out.dtype == torch.int32
            and out.device == q.device
            and out.is_contiguous()
            and math.isfinite(r)
            and 0 <= r <= torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "ball_into requires nonempty support, finite nonnegative FP32 radius, K in [1,32], and contiguous CUDA int32 output (Q,K)"
            )
        self.stats["ball_calls"] += 1
        if not len(q):
            return
        # Rows are retained privately; caller writes cannot alter a cached result.
        for saved_x, saved_q, radius, cached in self.memo:
            if (
                radius == r
                and len(x) == len(saved_x)
                and len(q) <= len(saved_q)
                and self._equal(x, saved_x)
                and self._equal(q, saved_q[: len(q)])
            ):
                out.copy_(cached[: len(q), :k])
                self.stats["ball_hits"] += 1
                return
        cap = 32
        result = torch.zeros((len(q), cap), device=q.device, dtype=torch.int32)
        if len(x) <= self.threshold:
            owned_x = self._snapshot(x)
            self.ball_reference(
                1, len(x), len(q), r, cap, q[None], owned_x[None], result[None]
            )
        else:
            tree = self._tree(x)
            owned_x = tree.x[: len(x)]
            tree.ball(q, len(x), r, cap, result)
        self.memo.append((owned_x, self._snapshot(q), r, result))
        self.stats["cache_bytes"] += result.numel() * result.element_size()
        out.copy_(result[:, :k])

    def nn_into(self, q, x, d, ids):
        self._check(x)
        self._check(q)
        if not (
            len(x) >= 3
            and d.shape == ids.shape == (len(q), 3)
            and d.dtype == torch.float32
            and ids.dtype == torch.int32
            and d.device == ids.device == q.device
            and d.is_contiguous()
            and ids.is_contiguous()
        ):
            raise ValueError(
                "nn_into requires at least 3 supports and contiguous CUDA FP32 distances/int32 IDs, both (Q,3)"
            )
        self.stats["nn_calls"] += 1
        if not len(q):
            return
        if len(x) <= self.threshold:
            self.nn_reference(1, len(q), len(x), q[None], x[None], d[None], ids[None])
        else:
            self._tree(x).query(q, len(x), d, ids)

    def close(self, collect=False):
        if self.closed:
            return
        if (
            torch.cuda.current_device() != self.device
            or torch.cuda.current_stream().cuda_stream != self.stream
        ):
            raise ValueError("Close the session on its original device and stream")
        for _, tree, _ in self.trees:
            if collect:
                self.stats["fallbacks"] += int(tree.fallback.item())
            tree.close()
        self.trees.clear()
        self.memo.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
