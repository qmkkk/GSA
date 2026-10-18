# SPDX-License-Identifier: Apache-2.0
"""ctypes binding to the exact CUDA BVH implementation."""

import ctypes
import torch
import math
from ._library import load

lib = load("gsa_bvh")
P = ctypes.c_void_p
I = ctypes.c_int
_pool_device = None
lib.build_bvh.argtypes = [P, I, I, I, ctypes.POINTER(P), P]
lib.build_bvh.restype = I
lib.query_bvh.argtypes = [P, P, I, I, I, P, P, P, P]
lib.query_bvh.restype = I
lib.query_ball_bvh.argtypes = [P, P, I, I, I, ctypes.c_float, I, P, P, P]
lib.query_ball_bvh.restype = I
lib.free_bvh.argtypes = [P, I, P]
lib.free_bvh.restype = I
lib.pool_memory.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
lib.pool_memory.restype = I
lib.pool_reset_peak.argtypes = []
lib.pool_reset_peak.restype = I
lib.set_pool_mode.argtypes = [I]
lib.set_pool_mode.restype = I


def check(e):
    if e:
        raise RuntimeError(f"BVH error {e}")


def pool_memory():
    a = ctypes.c_uint64()
    b = ctypes.c_uint64()
    check(lib.pool_memory(ctypes.byref(a), ctypes.byref(b)))
    return {
        "native_pool_used_peak_bytes": a.value,
        "native_pool_reserved_peak_bytes": b.value,
    }


class Tree:
    def __init__(self, x, prefix=False, leaf=16):
        global _pool_device
        if not (
            x.is_cuda
            and x.is_contiguous()
            and x.dtype == torch.float32
            and x.ndim == 2
            and x.shape[1] == 3
            and len(x) > 0
            and x.device.index == torch.cuda.current_device()
            and isinstance(leaf, int)
            and leaf >= 1
        ):
            raise ValueError(
                "Tree requires nonempty contiguous CUDA FP32 (N,3) on the current device and a positive leaf size"
            )
        if _pool_device is not None and _pool_device != x.device.index:
            raise ValueError(
                "The native allocator supports one CUDA device per process"
            )
        _pool_device = x.device.index
        self.stream = torch.cuda.current_stream().cuda_stream
        self.x = x
        self.prefix = prefix
        self.ptr = P()
        self.fallback = torch.zeros(1, device=x.device, dtype=torch.int32)
        check(
            lib.build_bvh(
                x.data_ptr(),
                len(x),
                int(prefix),
                leaf,
                ctypes.byref(self.ptr),
                torch.cuda.current_stream().cuda_stream,
            )
        )

    def level(self, m):
        if not self.prefix:
            return -1
        size = len(self.x)
        sizes = []
        for _ in range(4):
            sizes.append(size)
            size //= 4
        return sizes.index(m)

    def _validate_query(self, q, m, idx, width):
        if not (
            self.ptr
            and torch.cuda.current_device() == self.x.device.index
            and torch.cuda.current_stream().cuda_stream == self.stream
            and q.device == self.x.device
            and q.is_contiguous()
            and q.dtype == torch.float32
            and q.ndim == 2
            and q.shape[1] == 3
            and idx.device == q.device
            and idx.is_contiguous()
            and idx.dtype == torch.int32
            and idx.shape == (len(q), width)
            and 1 <= m <= len(self.x)
        ):
            raise ValueError(
                "Invalid query buffers, support length, or tree lifetime/device/stream"
            )

    def query(self, q, m, d, idx):
        self._validate_query(q, m, idx, 3)
        if not (
            m >= 3
            and d.device == q.device
            and d.is_contiguous()
            and d.dtype == torch.float32
            and d.shape == (len(q), 3)
        ):
            raise ValueError(
                "3NN requires at least 3 supports and FP32 (Q,3) distances"
            )
        if not len(q):
            return d, idx
        check(
            lib.query_bvh(
                self.ptr,
                q.data_ptr(),
                len(q),
                m,
                self.level(m),
                d.data_ptr(),
                idx.data_ptr(),
                self.fallback.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
        )
        return d, idx

    def ball(self, q, m, r, ns, idx):
        self._validate_query(q, m, idx, ns)
        if not (
            isinstance(ns, int)
            and 1 <= ns <= 32
            and math.isfinite(r)
            and 0 <= r <= torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "Radius query requires K in [1,32] and finite nonnegative FP32 radius"
            )
        if not len(q):
            return idx
        check(
            lib.query_ball_bvh(
                self.ptr,
                q.data_ptr(),
                len(q),
                m,
                self.level(m),
                r,
                ns,
                idx.data_ptr(),
                self.fallback.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
        )
        return idx

    def close(self):
        if self.ptr:
            if (
                torch.cuda.current_device() != self.x.device.index
                or torch.cuda.current_stream().cuda_stream != self.stream
            ):
                raise ValueError("Close the tree on its original device and stream")
            check(lib.free_bvh(self.ptr, 1, torch.cuda.current_stream().cuda_stream))
            self.ptr = P()
