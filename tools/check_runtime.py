"""Check board dependencies and RKNN checksums without changing the system."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

root=Path(__file__).resolve().parent.parent
for module,package in [('cv2','opencv-python'),('numpy','numpy'),('torch','torch'),
                       (None,'torchvision'),('ultralytics','ultralytics'),
                       ('lap','lap'),('rknnlite.api','rknn-toolkit-lite2')]:
    # The NPU/ByteTrack path needs torchvision metadata, not its datasets/operators.
    if module:
        importlib.import_module(module)
    print(f'{package}: {importlib.metadata.version(package)}')
manifest=json.loads((root/'models/manifest.json').read_text())
for name,info in manifest['models'].items():
    model=root/'models'/name
    if hashlib.sha256(model.read_bytes()).hexdigest()!=info['sha256']:
        raise SystemExit(f'Model checksum mismatch: {name}')
    print(f'{name}: checksum OK')
print('Environment and model files are ready. NPU initialization occurs when analysis starts.')
