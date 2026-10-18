# GSA: Geometry-State Reuse for Exact GPU Queries

**GSA** accelerates radius and three-nearest-neighbor queries in hierarchical point-cloud networks by sharing spatial indices and reusing eligible query results. It preserves the original support-point order and explicit FP32 query semantics.

This repository contains the CUDA implementation, Python interfaces, independent correctness tests, and per-case experimental measurements for **GSA: Geometry-State Reuse for Exact GPU Queries in Hierarchical Point-Cloud Networks**.

**Authors:** Tianyu He, Qixuan Yan, Yunhao Ma, Rundong Zhang  
College of Information Science and Engineering, Northeastern University, Shenyang, China.

## What GSA shares

1. **Spatial index:** one BVH carries bounds for four ordered support prefixes. Radius and 3NN queries share its topology while using the bounds for their support set.
2. **Radius results:** a private result row can serve a shorter query prefix and a smaller neighbor count when the complete ordered support, coordinates, and radius match.
3. **Execution lifetime:** index and cache state live within a forward/session. `QuerySession` owns coordinate snapshots and checks equality before reuse; the fixed-graph adapter uses the validated hierarchy contract.

The BVH builder comes from [NVIDIA cuBQL](https://github.com/NVIDIA/cuBQL). The original-order fallback kernels come from [FlashFPS/OpenPoints](https://github.com/Yuzhe-Fu/FlashFPS). Dependency versions and attribution are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Results at a glance

The recorded experiments use PointNeXt-L and PointVector-L on S3DIS and ScanNet, with an RTX 4060 Laptop GPU. GSA keeps the FlashFPS75 sampling configuration and pretrained weights fixed.

| Comparison | Cohort | Network-forward speedup |
|---|---|---|
| Strict-FP32 FlashFPS75 | 760 scene/model cases | 1.80–2.04× across four dataset/model groups |
| OrderedSIMD query baseline | 96 paired input cases | 1.079–1.185× |
| Sorted torchbvh baseline | Same 96 paired input cases | 1.074–1.187× |

The 760-case study checks every voxel-pass logit tensor and merged prediction against the strict-FP32 reference. Group speedups are geometric means of paired ratios. The strong-baseline data separately report logit differences, prediction differences, and mIoU for each baseline's arithmetic. [Evaluation details](docs/evaluation.md) define the timing boundaries, cohorts, and memory metric.

## Quick start

Use an existing CUDA-enabled PyTorch installation and CUDA toolkit. On Windows, run the build in an **x64 Visual Studio Native Tools** shell. The tested environment is Windows x64, Python 3.13.5, PyTorch 2.9.0+cu128, CUDA compiler 12.6, and Visual Studio 2022.

```sh
git clone https://github.com/qmkkk/GSA.git
cd GSA
python -m pip install --no-deps --no-build-isolation -e .
python scripts/build.py --arch 89
python tests/check_queries.py
python tests/check_integration.py
python examples/query_session.py
```

`89` is the compute capability of the tested GPU; choose the value for your GPU. The build uses `--fmad=false --ftz=false` to retain the query arithmetic. `--nvcc` can select a particular CUDA compiler. A Linux shared-library build path is supplied; the release validation platform is Windows. Runtime prerequisites are PyTorch and NumPy, with setuptools for the editable install. No dataset is needed for the examples or query tests.

## Use the query interface

```python
import torch
from gsa import QuerySession

x = torch.randn(4096, 3, device="cuda", dtype=torch.float32)
q = x[:128].clone()
ids = torch.empty((128, 16), device="cuda", dtype=torch.int32)

with torch.inference_mode(), QuerySession() as session:
    session.ball_into(q, x, 0.4, 16, ids)
    print(session.stats)
```

Use contiguous finite FP32 `(N,3)` coordinates, one CUDA device and stream per process/session, and sequential calls. Radius query returns the first `K` qualifying original IDs (`K <= 32`); 3NN returns squared distances and original IDs. The implementation targets inference. See [API semantics](docs/api.md) and [network integration](docs/integration.md).

## Experimental data and recomputation

The `data/` directory includes the full 760-case matrix, 96 strong-baseline cases, 24 ablation cases, 96 QuerySession network cases, 144 resource-measurement process records, correctness records, summaries, and figure data tables. Each cohort retains every measured case and every recorded repeat.

```sh
python scripts/verify_results.py
```

This CPU-only command validates file hashes and recomputes main accuracy/speed results, strong-baseline ratios, ablations, resource results, and scale bins. It writes a compact comparison table to `results/recomputed.csv`. See [data/README.md](data/README.md) for the schema and [dataset sources](docs/datasets.md) for source point clouds and pretrained checkpoints.

## Repository guide

| Path | Contents |
|---|---|
| `gsa/` | QuerySession, low-level bindings, fixed-graph adapter |
| `csrc/` | Exact BVH radius/3NN kernels and original-order reference queries |
| `tests/` | Independent NumPy oracle and integration checks |
| `examples/` | Dataset-free usage example |
| `scripts/` | Native build, numerical recomputation, cluster statistics |
| `data/` | Per-case measurements, summaries, plotting tables, asset checksums |
| `docs/` | API, integration, evaluation, and data preparation guidance |
| `third_party/` | Pinned cuBQL headers and original license |

## Citation

Use [CITATION.cff](CITATION.cff) for the software citation. Please also cite the upstream methods and datasets used in your experiments. The citation describes this software release; publication metadata can be added when available.

## License

[Apache-2.0](LICENSE) covers the GSA-authored implementation, documentation, and experimental measurement tables. Third-party code retains its original license, as listed in [THIRD_PARTY.md](THIRD_PARTY.md). Source datasets and pretrained checkpoints are obtained under their publishers' terms.
