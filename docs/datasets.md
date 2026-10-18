# Source datasets and checkpoints

The repository publishes experimental **measurement data** directly under `data/`. For network execution, obtain the source point clouds and pretrained weights from their original publishers, then verify them against `data/assets/` where the same prepared files are used.

## S3DIS and ScanNet

Follow the dataset preparation instructions in the pinned [FlashFPS/OpenPoints README](https://github.com/Yuzhe-Fu/FlashFPS/blob/f32a4b7886501873f2d7d380dc84f07ae77492df/FlashFPS-Openpoints/README.md) and its referenced [PointNeXt documentation](https://github.com/guochengqian/PointNeXt). The full-scene measurements use S3DIS Area 5 and ScanNet validation. ScanNet access and terms are managed by the [ScanNet project](https://github.com/ScanNet/ScanNet).

`data/assets/s3dis.json` identifies the 68 prepared room arrays. `data/assets/scannet.json` identifies the 312 prepared validation scans. Each entry provides its source-relative name, byte size, and SHA-256. These hashes refer to the prepared files used in the measurements; a different serialization can represent the same coordinates while producing a different file hash.

## Pretrained weights

Use the pretrained PointNeXt-L and PointVector-L weights linked in the [FlashFPS model instructions](https://github.com/Yuzhe-Fu/FlashFPS/blob/f32a4b7886501873f2d7d380dc84f07ae77492df/FlashFPS-Openpoints/README.md#1-download-models). The upstream [checkpoint collection](https://drive.google.com/drive/folders/1ChRWL8lk5bidxpub1OXAX1sAdZlFlFj-) provides both datasets. Four checksum entries in `data/assets/weights.json` identify the exact local names used for the measured checkpoints.

Load tensor checkpoints with `torch.load(..., weights_only=True)`, remove a `module.` key prefix if present, and require strict model-state loading. Preserve the upstream architecture and evaluation preprocessing. GSA operates on the radius/3NN calls after sampling; [integration.md](integration.md) gives the binding interface.

## Lightweight execution

`examples/query_session.py` and both test scripts generate deterministic synthetic coordinates, so core compilation and correctness can be checked immediately without dataset access or pretrained weights. `scripts/verify_results.py` recomputes the included measurement results on a CPU-only machine.
