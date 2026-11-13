"""File/USB/RTSP demo using the original proximity engine and real ByteTrack."""
import argparse
import os
# Tiny Kalman systems should not start a large BLAS thread pool per update.
os.environ['OPENBLAS_NUM_THREADS'] = os.environ.get('WAREHOUSE_BLAS_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('WAREHOUSE_BLAS_THREADS', '1')
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import config
from scripts.engine import ProximityAnalyzer, IMAGE_CRITICAL_RADIUS, IMAGE_CAUTION_MARGIN, IMAGE_CRITICAL_HYSTERESIS


def run(source, model, backend='rknn', output='outputs/demo', max_frames=0,
        meters_per_pixel=0.01, calibrated=False, callback=None, stop=None,
        risk_mode='image_zone', vehicle_extension=1.0, image_caution_ratio=IMAGE_CAUTION_MARGIN):
    out = Path(output)
    cv2.setNumThreads(2)
    out.mkdir(parents=True, exist_ok=True)
    capture_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if np.isfinite(fps) and fps > 0 else 30.0
    live = isinstance(capture_source, int) or str(source).startswith(('rtsp://', 'http://', 'https://', '/dev/video'))
    edge = writer = analyzer = trace = None
    n, latencies = 0, []
    started = time.perf_counter()
    try:
        if backend != 'pt':
            from scripts.edge_backend import EdgeModel
            edge = EdgeModel(model, backend, fps, config.CONFIDENCE_THRESHOLD)
        analyzer = ProximityAnalyzer(model, config.WAREHOUSE_CLASSES,
                    model=edge, meters_per_pixel=meters_per_pixel,
                    risk_mode=risk_mode,vehicle_extension=vehicle_extension,
                    image_caution_ratio=image_caution_ratio)
        ready = time.perf_counter()
        trace = (out/'frame-trace.jsonl').open('w',encoding='utf-8')
        while not (stop and stop.is_set()):
            ok, frame = cap.read()
            if not ok:
                if live: raise RuntimeError('Live source disconnected')
                break
            t = time.perf_counter() - ready if live else (n + 1) / fps
            annotated = analyzer.process_frame(frame, fps, timestamp_s=t)
            cv2.putText(annotated, 'reComputer RK3576 | YOLO11 + ByteTrack | ' + backend.upper(),
                        (16, frame.shape[0]-40), 0, .6, (255,255,255), 2)
            cv2.putText(annotated, 'DEMO | ' + ('IMAGE ZONE ALERTS - NOT METRIC DISTANCE' if risk_mode == 'image_zone'
                        else 'USER CALIBRATION' if calibrated else 'UNCALIBRATED DISTANCE / SPEED'),
                        (16, frame.shape[0]-15), 0, .55, (0,200,255), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(out/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Cannot create output video')
            writer.write(annotated)
            n += 1
            if edge: latencies.append(edge.last_inference_ms)
            active = [dict(forklift_id=p.forklift_id, person_id=p.person_id,
                           tier=p.tier, distance_m=round(p.last_distance_m,2) if risk_mode=='legacy' else None,
                           closing_mps=round(p.smoothed_closing_speed_mps,2) if risk_mode=='legacy' else None,
                           image_gap_ratio=p.image_gap_ratio, reason=p.alert_reason)
                      for p in analyzer.pairs.values() if p.last_seen_frame == analyzer.frame_idx]
            trace.write(json.dumps(dict(analyzer.last_observation,time_s=t,pairs=active))+'\n')
            if n in (1, 72, 120, 192): cv2.imwrite(str(out/f'frame-{n:04d}.jpg'), annotated)
            if callback:
                callback(annotated, {'frame': n, 'source_s': round(t,2), 'pairs': active,
                                    'events': analyzer.log_rows,
                                    'fps': round(n/max(time.perf_counter()-ready,1e-6),2)})
            if max_frames and n >= max_frames: break
        if n == 0: raise RuntimeError('Source contains no decodable frames')
        elapsed = time.perf_counter() - ready
        analyzer.flush_open_events()
        analyzer.save_log(out/'events.csv')
        steady = latencies[15:] or latencies
        summary = {
            'backend': backend, 'platform': platform.platform(), 'machine': platform.machine(),
            'model_sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            'frames': n, 'source_fps': fps, 'elapsed_s': round(elapsed,3),
            'pipeline_fps': round(n/elapsed,2),
            'startup_s': round(ready-started,3),
            'inference_ms_median_after_warmup': round(float(np.median(steady)),3) if steady else None,
            'inference_ms_p95_after_warmup': round(float(np.percentile(steady,95)),3) if steady else None,
            'warmup_frames': min(15, len(latencies)),
            'calibration': {'meters_per_pixel': meters_per_pixel, 'user_calibrated': calibrated},
            'risk_mode': risk_mode,
            'image_zone': {'horizontal_extension_box_widths':vehicle_extension,
                           'ellipse_horizontal_radius_box_widths':.5+1.9*vehicle_extension,
                           'ellipse_vertical_radius_box_heights':.4,
                           'critical_ellipse_radius':IMAGE_CRITICAL_RADIUS,
                           'person_zone_anchor':'closest_point_on_bbox_bottom_edge',
                           'critical_clear_hysteresis_radial_margin':IMAGE_CRITICAL_HYSTERESIS,
                           'caution_radial_margin':image_caution_ratio,
                           'clear_hysteresis_radial_margin':.15},
            'events': analyzer.log_rows,
            'video_timing': 'nominal source FPS; live recording playback may run faster than capture',
        }
        (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    finally:
        cap.release()
        if writer: writer.release()
        if analyzer:
            analyzer.flush_open_events()
            analyzer.save_log(out/'events.csv')
        if edge: edge.close()
        if trace: trace.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=config.DEFAULT_INPUT_VIDEO)
    p.add_argument('--model', default='models/warehouse_yolo11_rk3576_int8.rknn')
    p.add_argument('--backend', choices=['rknn','onnx','pt'], default='rknn')
    p.add_argument('--output', default='outputs/demo')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--meters-per-pixel', type=float, default=.01)
    p.add_argument('--calibrated', action='store_true')
    p.add_argument('--risk-mode', choices=['legacy','image_zone'],default='image_zone')
    p.add_argument('--vehicle-extension',type=float,default=1.0)
    p.add_argument('--image-caution-ratio',type=float,default=IMAGE_CAUTION_MARGIN)
    a = p.parse_args()
    if a.meters_per_pixel <= 0: p.error('--meters-per-pixel must be positive')
    print(json.dumps(run(a.source,a.model,a.backend,a.output,a.max_frames,
                         a.meters_per_pixel,a.calibrated,risk_mode=a.risk_mode,
                         vehicle_extension=a.vehicle_extension,image_caution_ratio=a.image_caution_ratio), indent=2))
