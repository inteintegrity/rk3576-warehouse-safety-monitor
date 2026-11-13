"""Prepare letterboxed calibration frames. Sample-video frames are demo-only."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from scripts.edge_backend import letterbox

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--video', default='data/test_clip_nearmiss.mp4')
p.add_argument('--output', default='data/calibration')
p.add_argument('--stride', type=int, default=10)
a = p.parse_args()
if a.stride < 1: p.error('stride must be positive')
out = Path(a.output)
out.mkdir(parents=True,exist_ok=True)
cap = cv2.VideoCapture(a.video)
paths, n = [], 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if n % a.stride == 0:
        image,_ = letterbox(frame)
        path = out/f'frame-{n:04d}.png'
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(path)
        paths.append(path.resolve().as_posix())
    n += 1
cap.release()
if not paths: raise RuntimeError('No calibration frames decoded')
(out/'dataset.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
print(f'{len(paths)} calibration frames; not an independent validation set')
