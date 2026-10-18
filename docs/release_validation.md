# Release validation

Version 0.1.0 was checked on the Windows/CUDA reference platform described in `data/environment.json`.

| Check | Result |
|---|---|
| Native library build from the distributed sources | Passed |
| Independent NumPy FP32 oracle and sequential state transitions | 94 checks passed |
| Fixed-graph prefix/scale output equivalence and cache reuse | Passed |
| Exception cleanup and invalid-buffer/radius rejection | Passed |
| PointNeXt-L and PointVector-L, S3DIS and ScanNet, both public execution interfaces | 8 full forward outputs matched the recorded strict-FP32 logit hashes |
| Numerical recomputation from the included measurements | 760 main cases, 96 strong-baseline cases, 96 session cases, 24 ablation cases, and 144 resource records passed |
| Input-data and checkpoint identification | SHA-256 manifests supplied |

Release checks validate source packaging, query semantics, integration, and numeric reproducibility. The performance measurements in `data/` are the original evaluation measurements under the protocols in [evaluation.md](evaluation.md). The public Python interfaces add descriptive argument validation and portable library loading around the same computational kernels.
