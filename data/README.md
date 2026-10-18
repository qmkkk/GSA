# Experimental measurement data

All files are UTF-8 JSON or CSV. Times retain their original floating-point values, confusion matrices retain integer counts, and output hashes retain the measured values. Cases are selected by complete experimental cohort. Machine paths, process identifiers, orchestration timestamps, and working-directory bookkeeping are omitted. `manifest.json` gives SHA-256 checksums for the public data files.

| Directory | Unit | Count |
|---|---|---|
| `full/` | Complete scene/model, including every voxel pass | 760 |
| `strong_baselines/` | Dataset/scene/model/input-seed case | 96 |
| `query_session/` | Same network-case unit, plus one 94-check oracle record | 96 + 1 |
| `ablations/` | Scene/model/input-seed case, all nine variants | 24 |
| `resources/first/` | Fresh process for one method/pair/block | 108 |
| `resources/warm/` | Fresh process with three warmed methods | 36 |
| `summaries/` | Full, strong, session, ablation, resource, scale summaries | 6 |
| `plot_tables/` | Numerical tables used by the manuscript figures | 9 |
| `assets/` | Source data and checkpoint filenames, sizes, hashes | 3 manifests |

## Main fields

`dataset`, `scene`, `model`, and `seed` identify an input case. `raw_n` and `voxel_n` describe source and network-input point counts. `methods` maps method IDs to measurements; names are defined in [evaluation.md](../docs/evaluation.md).

In full-scene records, each method contains `timings` (one entry per voxel pass), `confusion`, `prediction_sha256`, accuracy metrics, and evaluation-path time. `wall_ms` is host-observed synchronized forward time in milliseconds; `gpu_ms` is CUDA-event time in milliseconds. Fields ending in `_seconds` use seconds and fields ending in `_bytes` use bytes.

In strong-baseline, session, and ablation records, `methods[method]` is the complete list of measured repeats. Strong records additionally include `comparison_reference_rate`, exactness flags, numerical differences, and per-repeat confusion matrices. `source_data_sha256` identifies the prepared input file. Input and output hashes support equality checking without distributing raw point coordinates or checkpoints.

Resource records contain `measurements` and `profiles`; warm records distinguish `warmup` and `measured` phases. Preserve this distinction when recomputing latency. The data include startup and warm-up values as well as the samples used in the published aggregates.

Summaries carry aggregate metrics, per-case values, confidence intervals, and explicit metric definitions. `scale.json` groups all 760 main cases by actual network-input point count: <50k, 50k–100k, 100k–150k, and >=150k points.

## Recompute

From the repository root:

```sh
python scripts/verify_results.py
```

Source point clouds and pretrained weights are identified by `assets/` and obtained from the publishers linked in [datasets.md](../docs/datasets.md). Experimental measurements are covered by the root license; original datasets and checkpoints retain their own terms.
