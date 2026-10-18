# Third-party software

| Component | Version used | License | Distribution |
|---|---|---|---|
| [NVIDIA cuBQL](https://github.com/NVIDIA/cuBQL) | `20f9db19cdb160e2b83d8f05dfdb0437fe11fe24` | Apache-2.0 | Unmodified headers in `third_party/cuBQL`; original LICENSE retained |
| [FlashFPS / OpenPoints](https://github.com/Yuzhe-Fu/FlashFPS) | `f32a4b7886501873f2d7d380dc84f07ae77492df` | MIT | Radius and 3NN forward query bodies in `csrc/reference_queries.cu`; license in `licenses/FlashFPS-MIT.txt` |
| [torchbvh](https://github.com/Robh96/torchbvh) | `30d7f3091cdab4cb4cd321674e1358f39bf31b96` | See upstream | Evaluated baseline; source obtained upstream |
| [FastPoint](https://github.com/SNU-ARC/FastPoint) | `7bebca723d5ba87891a9f221e3f0cf96962fab50` | See upstream | Evaluated baseline; source obtained upstream |

GSA uses cuBQL's spatial-median builder and bottom-up refit pattern. GSA adds prefix-specific bounds, exact original-ID query handling, and execution-state reuse. The CUDA fallback kernels retain the original upstream computation; this distribution extracts the radius and 3NN kernels and supplies portable C exports. FlashFPS credits OpenPoints, PointNeXt, FractalCloud, QuickFPS, and FastPoint in its upstream repository.

PyTorch and NumPy are installed dependencies governed by their own licenses. NVIDIA CUDA and the host compiler are user-installed build tools. GPU runtime binaries are built locally from the supplied source.
