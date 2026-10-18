# SPDX-License-Identifier: Apache-2.0
"""Compile the native query libraries with explicit FP32 arithmetic."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", default="89", help="CUDA compute capability without dot, e.g. 89"
    )
    parser.add_argument("--nvcc", default=shutil.which("nvcc"), help="Path to nvcc")
    args = parser.parse_args()
    if not args.arch.isdigit():
        parser.error("--arch must contain digits")
    if not args.nvcc:
        parser.error("nvcc was not found; activate your CUDA toolkit or pass --nvcc")
    root = Path(__file__).resolve().parents[1]
    output = root / "gsa/_native"
    output.mkdir(exist_ok=True)
    common = [
        args.nvcc,
        "-O3",
        "--fmad=false",
        "--ftz=false",
        "-std=c++17",
        f"-arch=sm_{args.arch}",
        "--extended-lambda",
        "--shared",
    ]
    if os.name == "nt":
        if not shutil.which("cl"):
            parser.error(
                "Run in an x64 Visual Studio Native Tools shell with cl.exe on PATH"
            )
        common += ["-Xcompiler=/O2,/utf-8,/EHsc,/DNDEBUG"]
        suffix = ".dll"
    else:
        common += ["-Xcompiler=-fPIC"]
        suffix = ".so"
    for name in ["gsa_bvh", "reference_queries"]:
        command = common + [
            "-I" + str(root / "third_party/cuBQL"),
            str(root / "csrc" / (name + ".cu")),
            "-o",
            str(output / (name + suffix)),
        ]
        subprocess.run(command, cwd=output, check=True)
    print("Built GSA. Next: python tests/check_queries.py")


if __name__ == "__main__":
    main()
