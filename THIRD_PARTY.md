# Third-party attribution

This release reuses the original warehouse model and proximity engine from
https://github.com/sounak-sarkar-gwc/Forklift-pedestrian-proximity (snapshot 35153b4).
The downloaded original snapshot does not contain a LICENSE file. This release
does not assign a new license to that source, media, or derived model weights.
Confirm upstream redistribution terms before a public release.

`scripts/engine.py` adapts its tracking/pair/event logic. Changes include the
RKNN adapter contract, configurable smoothing, elapsed timestamps, empty CSV
support, track cleanup, native JSON numbers, projected image regions, closest
pedestrian bottom-edge membership and danger/caution hysteresis.
The final package omits the earlier mock detector and uses the trained warehouse
model for both classes.

`scripts/rknn_postprocess.py` and `tools/convert_rknn.py` adapt Rockchip's
https://github.com/airockchip/rknn_model_zoo examples/yolo11 implementation.
See `third_party/RKNN-Model-Zoo-LICENSE` (Apache-2.0). Changes replace Torch DFL
with NumPy, remove COCO-only assumptions, expose representative calibration,
and check conversion failures.

`tools/export_onnx.py` follows the nine-output head in
https://github.com/airockchip/ultralytics_yolo11/blob/main/ultralytics/nn/modules/head.py
(identified upstream as AGPL-3.0). Ultralytics ByteTrack is used through the
installed Ultralytics package; preserve its upstream licensing and notices.

The INT8 calibration set used 24 frames from the original sample, which was also
used to train the supplied checkpoint. This is a reproduction demo, not an
independent evaluation. No generated demo video, third-party media, PT checkpoint,
training dataset or ONNX intermediate is included in this repository.
