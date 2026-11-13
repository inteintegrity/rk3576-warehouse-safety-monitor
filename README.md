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

## Install on the board

Use Python 3.11 on ARM64. Install the official matching
[RKNN Toolkit Lite2 2.3.2 wheel](https://github.com/airockchip/rknn-toolkit2)
for the board's Python version. Reuse a working RKNN runtime when available.
The tested environment used Torch 2.2.0 and torchvision 0.17.0; ByteTrack runs on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.0 torchvision==0.17.0
python -m pip install -r requirements.txt
# Install the official rknn_toolkit_lite2 2.3.2 ARM64 wheel for your Python version.
```

On an existing board environment with OpenCV, Torch and Lite2 already installed,
activate that environment and install only missing dependencies. Do not install
desktop PyTorch wheels or the full conversion toolkit on the board unnecessarily.

## Run

```bash
# Download the unchanged upstream sample; media is not stored in this repository.
python tools/fetch_sample.py

# Real NPU inference, final red/yellow/green alert rules.
python scripts/run_demo.py --backend rknn \
  --model models/warehouse_yolo11_rk3576_int8.rknn \
  --output outputs/demo

# Optional local dashboard.
python server.py --host 0.0.0.0 --port 8088
```

Open `http://<board-ip>:8088` on your trusted LAN. The single-operator dashboard
provides MJPEG preview, current pairs, and completed-run downloads. It has no
authentication and is intended for a trusted demo network.

For another source, add `--source /path/to/video.mp4`, `--source 0` for USB, or
`--source rtsp://...`. Only file-source inference was verified during this reproduction.
Live recording uses nominal camera FPS, so playback timing can differ from capture time.

Each run writes `annotated.mp4`, `events.csv`, `summary.json`, and
`frame-trace.jsonl` under the selected output directory. The live overlay uses
lightweight zone outlines and red danger boxes.

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
python tools/render_halo.py --run outputs/demo \
  --output outputs/showcase/warehouse-safety-master.mp4

ffmpeg -y -i outputs/showcase/warehouse-safety-master.mp4 \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart \
  outputs/showcase/warehouse-safety.mp4
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
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --basetemp .test-tmp
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
