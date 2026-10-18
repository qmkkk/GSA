# SPDX-License-Identifier: Apache-2.0
"""Load only shared libraries built into this package."""

import ctypes
import os
from pathlib import Path

_DLL_DIRS = []
if os.name == "nt":
    import torch

    for path in [
        Path(torch.__file__).parent / "lib",
        Path(os.environ.get("CUDA_PATH", ".")) / "bin",
    ]:
        if path.is_dir():
            _DLL_DIRS.append(os.add_dll_directory(str(path.resolve())))


def load(name):
    suffix = ".dll" if os.name == "nt" else ".so"
    path = Path(__file__).resolve().parent / "_native" / (name + suffix)
    if not path.is_file():
        raise RuntimeError(
            "Compile GSA first: python scripts/build.py --arch 89 (use your GPU architecture)"
        )
    return ctypes.CDLL(str(path))
