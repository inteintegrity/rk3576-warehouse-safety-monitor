"""Single-operator LAN demo dashboard; standard library HTTP + MJPEG.

Run: python server.py --host 0.0.0.0 --port 8088
No cloud service, telemetry, or extra dashboard dependencies.
"""
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = os.environ.get('WAREHOUSE_BLAS_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('WAREHOUSE_BLAS_THREADS', '1')
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import cv2
from scripts.run_demo import run
from scripts.engine import IMAGE_CAUTION_MARGIN

ROOT = Path(__file__).resolve().parent
state = {'running': False, 'frame': 0, 'fps': 0, 'pairs': [], 'events': [], 'error': None}
lock = threading.RLock()
changed = threading.Condition(lock)
stop = threading.Event()
jpeg = None
sequence = 0
settings = None


def update(frame, data):
    global jpeg, sequence
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    with changed:
        state.update(data)
        if ok: jpeg = encoded.tobytes()
        sequence += 1
        changed.notify_all()


def worker(source):
    try:
        result = run(source, settings.model, settings.backend, ROOT/'outputs/dashboard',
                     meters_per_pixel=settings.meters_per_pixel,
                     calibrated=settings.calibrated, callback=update, stop=stop,
                     risk_mode=settings.risk_mode,vehicle_extension=settings.vehicle_extension,
                     image_caution_ratio=settings.image_caution_ratio)
        with lock: state.update(events=result['events'], summary=result)
    except Exception as exc:
        with lock: state['error'] = str(exc)
    finally:
        with changed:
            state['running'] = False
            changed.notify_all()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, status, data, mime='application/json'):
        content = json.dumps(data).encode() if mime == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self.send(200, (ROOT/'web/index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/status':
            with lock: return self.send(200, dict(state, backend=settings.backend,
                                                 calibrated=settings.calibrated))
        if path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last = -1
            try:
                while True:
                    with changed:
                        changed.wait_for(lambda: sequence != last, timeout=1)
                        data, last = jpeg, sequence
                    if data:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                         str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                    else:
                        time.sleep(.1)
            except (BrokenPipeError, ConnectionResetError):
                return
        names = {'/download/video':('annotated.mp4','video/mp4'),
                 '/download/csv':('events.csv','text/csv'),
                 '/download/json':('summary.json','application/octet-stream')}
        if path in names:
            name,mime = names[path]
            with lock:
                if state['running']: return self.send(409, {'error':'Wait for the run to finish'})
            file = ROOT/'outputs/dashboard'/name
            if file.exists(): return self.send(200,file.read_bytes(),mime)
        self.send(404, {'error':'Not found'})

    def do_POST(self):
        global jpeg, sequence
        if self.path == '/api/stop':
            stop.set()
            return self.send(200, {'stopping':True})
        if self.path != '/api/start': return self.send(404, {'error':'Not found'})
        try:
            length = int(self.headers.get('Content-Length',0))
            if not 0 < length <= 4096: raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            source = str(payload.get('source','data/test_clip_nearmiss.mp4')).strip()
            if not source: raise ValueError('Source is required')
        except (ValueError,TypeError) as exc:
            return self.send(400, {'error':str(exc)})
        with lock:
            if state['running']: return self.send(409, {'error':'Analysis already running'})
            state.clear()
            state.update(running=True,frame=0,fps=0,pairs=[],events=[],error=None)
            jpeg = None
            sequence += 1
            stop.clear()
            # Remove previous artifacts only by the three known file names.
            for name in ('annotated.mp4','events.csv','summary.json'):
                (ROOT/'outputs/dashboard'/name).unlink(missing_ok=True)
            threading.Thread(target=worker,args=(source,),daemon=True).start()
        self.send(202, {'started':True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8088)
    p.add_argument('--backend',choices=['rknn','onnx','pt'],default='rknn')
    p.add_argument('--model',default='models/warehouse_yolo11_rk3576_int8.rknn')
    p.add_argument('--meters-per-pixel',type=float,default=.01)
    p.add_argument('--calibrated',action='store_true')
    p.add_argument('--risk-mode',choices=['legacy','image_zone'],default='image_zone')
    p.add_argument('--vehicle-extension',type=float,default=1.0)
    p.add_argument('--image-caution-ratio',type=float,default=IMAGE_CAUTION_MARGIN)
    settings = p.parse_args()
    if settings.meters_per_pixel <= 0: p.error('meters-per-pixel must be positive')
    print(f'Dashboard: http://{settings.host}:{settings.port}',flush=True)
    try:
        ThreadingHTTPServer((settings.host,settings.port),Handler).serve_forever()
    except KeyboardInterrupt:
        stop.set()
