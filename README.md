# Warehouse Safety Monitor on reComputer RK3576

YOLO11n detection on the **RK3576 NPU**, real ByteTrack tracking, and forklift–pedestrian
proximity alerts. The demo uses green monitoring, yellow caution, and red **DANGER**
states, with a separate operator label. A reference-style offline renderer adds
glowing floor zones to actual inference results.

This is the cleaned release: final runtime source, reproducibility tools, tests,
and **both RK3576 RKNN models**. Development snapshots, upstream repository copies,
PT/ONNX intermediates, datasets, videos, logs, credentials, and generated outputs
are excluded.

## Hardware and pipeline

```text
Video / USB / RTSP
  → letterbox 640×640 RGB
  → YOLO11n RKNN / RK3576 NPU
  → CPU DFL + NMS
  → CPU ByteTrack
  → operator exclusion + projected-zone alerts
  → annotated video + CSV + JSON + per-frame trace
```

Tested on Seeed Studio reComputer RK3576 with Debian 12, kernel
`6.1.115-vendor-seeed-rk3576`, Toolkit2/Lite2 2.3.2, Runtime 2.3.0 and driver 0.9.8.
The final 240-frame INT8 run measured 10.52 FPS for the complete pipeline and
25.551 ms median NPU API inference after warmup. These are different measurements;
24 FPS showcase playback follows the original source timeline.

## Included models

| File | Precision | Target | Classes |
|---|---|---|---|
| `models/warehouse_yolo11_rk3576_int8.rknn` | INT8 | RK3576 | `0=forklift`, `1=person` |
| `models/warehouse_yolo11_rk3576_fp16.rknn` | FP16 | RK3576 | `0=forklift`, `1=person` |

Checksums and input information are in [models/manifest.json](models/manifest.json).
These models are converted from the original warehouse checkpoint, not stock COCO
weights. Stock COCO weights do not contain a forklift class.

## Quick start on the board

Use ARM64 Linux with Python 3.11 and a working RK3576 NPU driver/runtime.
The launcher uses a project-local `.venv`, an activated environment, or the
explicit `WAREHOUSE_PYTHON` interpreter. It does not replace system libraries.

```bash
git clone https://github.com/inteintegrity/rk3576-warehouse-safety-monitor.git
cd rk3576-warehouse-safety-monitor
bash run.sh setup
bash run.sh web
```

`setup` installs the Python dependencies and official RKNN Lite2 2.3.2 package,
verifies model hashes, and downloads the unchanged upstream sample. Fresh
installations also need disk space for PyTorch. Existing working environments
can be reused by activating them before running the script.

Open `http://<board-ip>:8088` on your trusted LAN and select **Start analysis**.
Press Ctrl+C in the terminal to stop the server. The dashboard is a
single-operator LAN demo without authentication.

Other commands:

```bash
bash run.sh video
bash run.sh video --source /path/to/video.mp4
bash run.sh web --port 8089
bash run.sh check
bash run.sh test
```

The `video` command saves `annotated.mp4`, `events.csv`, `summary.json`, and
`frame-trace.jsonl` under `outputs/demo`. File-source inference was verified;
USB/RTSP camera capture still needs validation with the intended devices.
`bash run.sh --help` lists available commands.

## Alert rules

The default mode uses a projected **image-space ellipse**, centered on the bottom
of the forklift box. Its horizontal radius is 2.4 forklift-box widths, and its
vertical radius is 0.4 box heights. Classification uses the point on the pedestrian's
bbox bottom edge closest to the forklift center:

- **DANGER / CRITICAL:** enters the inner radius 0.85; clears beyond 0.93.
- **CAUTION:** enters radius 1.05; clears beyond 1.20.
- **Monitoring:** outside the warning region. The outer green display ring is radius 1.35.

The driver is excluded using the original operator heuristic. Region sizes are
conservative demo settings, not physical distances. The detector's box can omit
forks/load, so the inner region was enlarged to warn earlier. At approximately
5 and 8 seconds in the sample, the revised board run raises red CRITICAL alerts.
The legacy uncalibrated center-distance algorithm remains available with
`--risk-mode legacy` for comparison.

## Render the glowing video

Copy the original sample and the board's completed output directory to a desktop
machine with Pillow and OpenCV. Run from this repository root:

```bash
bash run.sh render outputs/demo
```

The glow is offline presentation work. Boxes, identities, warning states, and
timestamps come from actual board traces; warnings are not inserted at preset times.
No 1080p live glow rendering performance claim is made.

## Optional model reproduction

Obtain the original `best.pt` checkpoint from the credited source below, then export
on a desktop with PyTorch and ONNX. The export adapter follows Rockchip's nine-output
YOLO11 detect head. On a separate Linux Toolkit2 2.3.2 environment:

```bash
python tools/export_onnx.py --weights /path/to/best.pt
python tools/convert_rknn.py
python tools/prepare_calibration.py
python tools/convert_rknn.py --dataset data/calibration/dataset.txt \
  --output models/warehouse_yolo11_rk3576_int8.rknn
```

PT/ONNX files and calibration images are deliberately gitignored. The original
sample was used both in upstream training and in this demo's INT8 calibration;
it is not an independent accuracy test.

## Validation

```bash
bash run.sh test
```

Eight regression checks cover operator exclusion, DFL equivalence, preprocessing,
empty detections/logging, elapsed-time closing speed, missing inputs, red alerts
at the near-miss frames, and tier progression with hysteresis.

## Attribution and limitations

- Original warehouse checkpoint and analytics:
  [sounak-sarkar-gwc/Forklift-pedestrian-proximity](https://github.com/sounak-sarkar-gwc/Forklift-pedestrian-proximity), snapshot `35153b4`.
- NPU decoding and conversion reference:
  [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11), snapshot `837db8e`.
- Export head reference:
  [airockchip/ultralytics_yolo11](https://github.com/airockchip/ultralytics_yolo11/blob/main/ultralytics/nn/modules/head.py).
- Tracking: [Ultralytics ByteTrack](https://github.com/ultralytics/ultralytics).

See [THIRD_PARTY.md](THIRD_PARTY.md). The original source snapshot had no LICENSE
file, so no new blanket license is asserted for that source or the derived weights.
The sample is fetched from its original location rather than redistributed here.
Projected image zones are a visualization heuristic, not calibrated collision
envelopes or a safety-certified monitoring system.
