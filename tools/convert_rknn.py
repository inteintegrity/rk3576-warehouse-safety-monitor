"""RK3576 conversion, adapted from RKNN Model Zoo yolo11/python/convert.py.

Rockchip source is Apache-2.0; see third_party/RKNN-Model-Zoo-LICENSE.
Modified: explicit calibration, target fixed to rk3576, checked return codes.
"""
import argparse
from pathlib import Path
from rknn.api import RKNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', default='models/warehouse_yolo11_rkopt.onnx')
    p.add_argument('--output', default='models/warehouse_yolo11_rk3576_fp16.rknn')
    p.add_argument('--dataset', help='Text file of representative image paths; enables INT8')
    a = p.parse_args()
    rknn = RKNN(verbose=False)
    try:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform='rk3576')
        for name, operation in (
            ('load', lambda: rknn.load_onnx(model=a.onnx)),
            ('build', lambda: rknn.build(do_quantization=bool(a.dataset), dataset=a.dataset)),
            ('export', lambda: rknn.export_rknn(a.output)),
        ):
            ret = operation()
            if ret != 0:
                raise RuntimeError(f'RKNN {name} failed: {ret}')
        print(Path(a.output).resolve())
    finally:
        rknn.release()


if __name__ == '__main__':
    main()
