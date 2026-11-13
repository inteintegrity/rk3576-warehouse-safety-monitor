"""Shared typography and palette for the final offline video renderer."""
from functools import lru_cache
from pathlib import Path
from PIL import ImageFont

BG=(13,24,29); PANEL=(22,38,44); WHITE=(239,247,245); MUTED=(145,169,174)
GREEN=(173,234,151); AMBER=(255,199,81); RED=(255,109,99); BLUE=(102,189,236)


@lru_cache(maxsize=64)
def font(size,bold=False):
    candidates=[f'C:/Windows/Fonts/segoeui{"b" if bold else ""}.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def text(draw,xy,value,size=24,color=WHITE,bold=False):
    draw.text(xy,str(value),font=font(size,bold),fill=color)
