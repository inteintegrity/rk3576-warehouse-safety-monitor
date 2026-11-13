"""Download the upstream demo video without committing media or training data."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL='https://raw.githubusercontent.com/sounak-sarkar-gwc/Forklift-pedestrian-proximity/35153b4/data/test_clip_nearmiss.mp4'
SHA256='dc42b3bb4d85dea174fb9e0f99900466ab792fa92d510e063c05393d3ff56db3'
out=Path(__file__).resolve().parent.parent/'data/test_clip_nearmiss.mp4'
if out.exists():
    if hashlib.sha256(out.read_bytes()).hexdigest()!=SHA256:
        raise SystemExit('Existing sample differs; refusing to overwrite it')
else:
    with urlopen(URL,timeout=60) as response:
        content=response.read()
    if hashlib.sha256(content).hexdigest()!=SHA256:
        raise SystemExit('Sample checksum mismatch')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(content)
print(out)
