# TurnPilot models

YOLO26 object detectors from Ultralytics, exported to ONNX for the detector in `../detector/`.

The weights and code are AGPL-3.0. Models trained with Ultralytics code are AGPL-3.0 too, so nothing that uses these files can go upstream.

| File | Source weights | Weights sha256 | ONNX size | ONNX sha256 |
| --- | --- | --- | --- | --- |
| `yolo26n.onnx` | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | 9.8 MB | `65b2e24edea581f11f74b0e598904dfe343356809c3ce6f3f90daeb16e4c4e4f` |
| `yolo26s.onnx` | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` | 38.1 MB | `a7679e007e7eccd6c61ad6cbe579cfc84fd893690e75866bde5c4af4fc6ce102` |

Both files were exported on 2026-10-26 with ultralytics 8.4.190, torch 2.14.1 and onnx 1.23.2 by `export_yolo26.py`:

- opset 17, `simplify=False`
- `nms=False`, which keeps YOLO26's NMS-free head. `output0` is `(1, 300, 6)`: x0, y0, x1, y1 in input pixels, then confidence and the COCO class id.
- `dynamic=True`, so one file serves every camera. `compile_detector.py` fixes the input size for each camera: 640x384 for a 1344x760 frame and 640x416 for a 1928x1208 frame. Through tinygrad, the outputs match onnxruntime on fixed-size exports to within 1.3e-4.
- The input is `images`, `(1, 3, H, W)` float32 RGB in the range 0 to 1.

The files are plain git blobs, not LFS, because `.gitattributes` exempts `openpilot/selfdrive/turnpilot/**`. upstream's `scripts/lint/lint.sh` limits non-LFS files under `openpilot/` to 120 KB, so these files fail its `check_added_large_files` step, as any detector would. Both are well under GitHub's 50 MB warning.

The compiled detectors (`detector_{w}x{h}_tinygrad.pkl` and `big_detector_{w}x{h}_tinygrad.pkl` for the chestnut) are built here by `../detector/compile_detector.py`. They are gitignored, except `big_detector_1344x760_tinygrad.pkl`: the car needs it and the installer can't build it, so it's a plain blob in git.

| File | Model | Built | Size | sha256 |
| --- | --- | --- | --- | --- |
| `big_detector_1344x760_tinygrad.pkl` | yolo26n, 640x384 input | 2026-10-27 on the Mac with the chestnut, `--retargetable`, tinygrad_repo 02e6981 | 20.5 MB | `f4f17f788e86b2b893472908df94ca5d10eca2e857389a3921c4d0eca82a0f11` |

Rebuild it with the chestnut attached and nothing else holding it:

```sh
PYTHONPATH=$PWD:$PWD/tinygrad_repo DEV=USB+AMD:LLVM python -m openpilot.selfdrive.turnpilot.detector.compile_detector --retargetable --sizes 1344x760
```

Measured on the chestnut on 2026-10-27, at 1344x760:

| Model | Alone | In modeld beside the big model | pkl |
| --- | --- | --- | --- |
| yolo26n | 5.9 ms | 7.9 ms; big model 25-27 ms; no frame drops even at every frame | 20.5 MB |
| yolo26s | 10.1 ms | 13.3 ms at every 2nd frame, no frame drops | 56.2 MB, over GitHub's 50 MB limit, so not used |

Both run in float32. The chestnut's tensor cores (RDNA4 WMMA) take half, bfloat16, fp8 and int8, not float32, so the compile context with `TC_OPT=2` changes nothing for these models (5.92 ms with and without it); a half-precision export is the way to use them.
