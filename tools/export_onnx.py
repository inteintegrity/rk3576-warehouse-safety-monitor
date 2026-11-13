"""Export the supplied two-class checkpoint using Rockchip's nine-output head.

Head adaptation follows airockchip/ultralytics_yolo11 Detect.forward (AGPL-3.0).
No retraining or weight replacement; DFL and NMS remain on the CPU.
"""
import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from ultralytics import YOLO


def rockchip_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv2[i](x[i]))
        cls = self.cv3[i](x[i]).sigmoid()
        outputs.extend((cls, cls.sum(1, keepdim=True).clamp(0, 1)))
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='models/warehouse_yolo11_v2/weights/best.pt')
    p.add_argument('--output', default='models/warehouse_yolo11_rkopt.onnx')
    a = p.parse_args()
    torch.set_num_threads(4)
    yolo = YOLO(a.weights)
    if yolo.names != {0: 'forklift', 1: 'person'}:
        raise ValueError(f'Wrong checkpoint classes: {yolo.names}')
    model = yolo.model.cpu().float().eval().fuse()
    head = model.model[-1]
    if head.nl != 3 or head.reg_max != 16:
        raise ValueError('Only the YOLO11 three-scale detect head is supported')
    head.forward = MethodType(rockchip_forward, head)
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        shapes = [list(x.shape) for x in model(dummy)]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), opset_version=12,
                      input_names=['images'],
                      output_names=[f'{kind}_{s}' for s in (8, 16, 32)
                                    for kind in ('boxes', 'classes', 'score_sum')],
                      dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(out)))
    out.with_suffix('.json').write_text(json.dumps({
        'classes': yolo.names, 'input': [1, 3, 640, 640], 'outputs': shapes,
        'normalization': 'RGB / 255', 'pad': 114,
        'head_reference': 'https://github.com/airockchip/ultralytics_yolo11/blob/main/ultralytics/nn/modules/head.py'
    }, indent=2), encoding='utf-8')
    print(out, shapes)


if __name__ == '__main__':
    main()
