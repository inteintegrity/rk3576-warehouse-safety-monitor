"""RKNN Lite2 / ONNX adapters; reuse Ultralytics ByteTrack and upstream risk engine."""
import time
from types import SimpleNamespace
import cv2
import numpy as np
from scripts import rknn_postprocess as pp


def letterbox(frame):
    h, w = frame.shape[:2]
    ratio = min(640 / h, 640 / w)
    nw, nh = round(w * ratio), round(h * ratio)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    resized = cv2.resize(frame, (nw, nh))
    image = cv2.copyMakeBorder(resized, top, 640 - nh - top,
                               left, 640 - nw - left, cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (nw / w, nh / h, left, top)


class EdgeModel:
    def __init__(self, path, backend='rknn', fps=24, confidence=0.35):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        self.Boxes = Boxes
        args = SimpleNamespace(track_high_thresh=confidence, track_low_thresh=0.1,
                               new_track_thresh=confidence, track_buffer=30,
                               match_thresh=0.8, fuse_score=True)
        # Prevent forklift/person identity swaps; keep original ByteTrack implementation.
        self.trackers = {cls: BYTETracker(args) for cls in (0, 1)}
        self.backend = backend
        self.last_inference_ms = 0.0
        if backend == 'rknn':
            from rknnlite.api import RKNNLite
            self.runtime = RKNNLite()
            if self.runtime.load_rknn(str(path)) != 0:
                raise RuntimeError(f'Cannot load RKNN model: {path}')
            if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                self.runtime.release()
                raise RuntimeError('Cannot initialize RK3576 NPU')
        elif backend == 'onnx':
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.runtime = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
        else:
            raise ValueError(backend)

    def detect(self, frame):
        image, (rx, ry, left, top) = letterbox(frame)
        start = time.perf_counter()
        if self.backend == 'rknn':
            outputs = self.runtime.inference(inputs=[image[None]], data_format=['nhwc'])
        else:
            tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255
            outputs = self.runtime.run(None, {self.runtime.get_inputs()[0].name: tensor})
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        if outputs is None or len(outputs) != 9:
            raise ValueError('Expected Rockchip nine-output YOLO11 head')
        for i, grid in enumerate((80, 40, 20)):
            for j, channels in enumerate((64, 2, 1)):
                if outputs[i*3+j].shape != (1, channels, grid, grid):
                    raise ValueError(f'Unexpected output shape: {outputs[i*3+j].shape}')
        boxes, classes, scores = pp.post_process(outputs)
        if boxes is None:
            return np.empty((0, 6), dtype=np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / rx
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ry
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.shape[0])
        return np.column_stack((boxes, scores, classes)).astype(np.float32)

    def track(self, frame, **kwargs):
        detections = self.detect(frame)
        tracks = []
        for cls, tracker in self.trackers.items():
            selected = detections[detections[:, 5] == cls]
            result = tracker.update(self.Boxes(selected, frame.shape[:2]))
            if len(result):
                tracks.append(result[:, :7])
        data = np.concatenate(tracks) if tracks else np.empty((0, 7), dtype=np.float32)
        # Existing analyzer uses .cpu().numpy(), so preserve that interface.
        class Array:
            def __init__(self, value): self.value = value
            def cpu(self): return self
            def numpy(self): return self.value
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=Array(data[:, :4]), id=Array(data[:, 4]), cls=Array(data[:, 6]))) ]

    def close(self):
        if self.backend == 'rknn':
            self.runtime.release()
