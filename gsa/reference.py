# SPDX-License-Identifier: Apache-2.0
"""Original-order brute-force CUDA query fallback from FlashFPS/OpenPoints."""

import ctypes
import torch
from ._library import load

lib = load("reference_queries")
P = ctypes.c_void_p
I = ctypes.c_int
lib.ball.argtypes = [I, I, I, ctypes.c_float, I, P, P, P, P]
lib.ball.restype = I
lib.nn3.argtypes = [I, I, I, P, P, P, P, P]
lib.nn3.restype = I


def call(name, *args):
    values = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
    error = getattr(lib, name)(*values, torch.cuda.current_stream().cuda_stream)
    if error:
        raise RuntimeError(f"{name}: CUDA error {error}")


def ball_query_wrapper(b, n, m, radius, k, q, x, out):
    call("ball", b, n, m, radius, k, q, x, out)


def three_nn_wrapper(b, n, m, q, x, distance, ids):
    call("nn3", b, n, m, q, x, distance, ids)
