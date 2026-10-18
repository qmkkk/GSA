# Evaluation protocol

## Reference environment

The measured platform is one NVIDIA RTX 4060 Laptop GPU with 8 GB memory, Windows x64, Python 3.13.5, PyTorch 2.9.0+cu128, CUDA compiler 12.6, and Visual Studio 2022. Neural inference uses batch size one and one CPU thread. Query correctness tests use four CPU threads for their NumPy/PyTorch harness. `data/environment.json` records dependency versions.

## Main full-scene evaluation

The full cohort contains 68 S3DIS Area-5 rooms and 312 ScanNet validation scans, each evaluated with PointNeXt-L and PointVector-L: 760 scene/model cases. Input seed is `20261070`. Multi-voxel evaluation covers every raw point and merges logits at their raw indices. Each scene warms up each method once before its first timed pass. There are 14,382 voxel passes and 43,146 measured forwards across three methods.

`flash` is the strict-FP32 FlashFPS75 reference; `scale` uses separate spatial indices at each support scale with eligible result reuse; `prefix` is fixed-graph GSA. Each voxel-pass tensor and merged prediction is compared with the strict-FP32 reference. Dataset-level mIoU is computed from summed confusion matrices, with zero IoU for a class whose union is zero.

Forward time covers sampling, neural layers, queries, index construction, cache maintenance, and index release. The separately reported evaluation-path time additionally includes scene preparation, voxel grouping, per-pass tensor preparation/upload, and download/merge. Checkpoint loading, integrity hashing, prediction serialization, reporting, and the initial unmeasured warm-up are outside that interval.

The primary speedup is the geometric mean of paired scene timing ratios. The stored 95% intervals use 10,000 physical-space cluster resamples: a room is one S3DIS cluster; all `sceneXXXX_*` scans belong to one ScanNet cluster. `scripts/clustered_statistics.py` implements this estimator. Repeated voxel passes remain within their scene. All cases, including the smallest-scene result, remain in the data.

## Strong baselines and QuerySession

These cohorts contain 24 physical scenes, two input seeds (`20261067`, `20261068`), and two models: 96 cases. Each configuration has five timed repetitions. A case first takes the median repetition time, seeds are combined within a scene in log-ratio space, and scene ratios are geometrically averaged. The same confirmation scenes appear in the main full-scene data; these cohorts answer different protocol questions on identified inputs.

Strong-baseline method identifiers distinguish arithmetic and sampling settings:

| Identifier | Meaning |
|---|---|
| `flash_strict_75` | FlashFPS75 with strict non-FMA query arithmetic |
| `flash_author_75`, `flash_author_50` | Author compilation path at the stated sampling setting |
| `scan_75` | OrderedSIMD scan queries at the 75 setting |
| `torchbvh_sorted_75`, `torchbvh_unsorted_75` | torchbvh query variants at the 75 setting |
| `prefix_75`, `prefix_50` | Fixed-graph GSA at the stated sampling setting |
| `fastpoint_author_o2`, `quick_author_o2` | Published sampling baselines with their recorded compilation path |

The strong-baseline records retain full-logit equality, maximum logit differences, prediction differences, and confusion matrices. Baselines with different arithmetic or sampling use their declared output reference. QuerySession records compare the original scan, fixed-graph GSA (`frozen`), and value-checked GSA (`session`).

## Ablations

The 24-case ablation cohort uses 12 development scenes with two models and five repetitions per method. `cache_only`, `ball_only`, and `nn_only` isolate components. `combined` enables the complete mechanism; `combined_nocache`, `combined_loose`, `scale`, and `default_allocator` vary reuse, bounds, topology sharing, and allocation respectively. These paired switches describe mechanism costs on the development cohort.

## First execution, warm execution, and memory

Six scenes (smallest, median, largest per dataset) and two models form 12 pairs. First execution/memory measurements use 108 fresh processes: 12 pairs × three methods × three blocks. A first forward begins with model weights, CUDA context, and inputs ready. Seven subsequent forwards retain the startup transition.

Warm measurements use 36 separate processes: 12 pairs × three blocks. Each method runs 30 warm-ups and seven measured forwards, with rotating method order. The reported warm latency is the median of each block's seven samples, then the median across three blocks. First, startup, warm-up, measured, and profiling outputs together yield 5,220 full-output checks.

The used-memory metric is `torch_peak_allocated_bytes + native_pool_used_peak_bytes`. It is a **conservative upper bound from separately observed allocator peaks**. Reserved-memory fields are reported separately; CUDA context and library residency are outside these allocator totals. Profiling records retain index construction, radius, 3NN, cache, and residual work with inclusive/exclusive definitions in the resource summary.

## Recompute versus rerun

`scripts/verify_results.py` recomputes numerical results from the included original measurement values, using only Python's standard library. The query and integration tests execute the released CUDA implementation. Full network experiments use the upstream neural implementations, checkpoints, preprocessing, and evaluation protocol described here and in [datasets.md](datasets.md); the public core API is integrated using [integration.md](integration.md).
