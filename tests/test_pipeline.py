from collections import deque
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from scripts.engine import ProximityAnalyzer, PairState, TrackState, _is_aboard, _image_zone_gap
from scripts.edge_backend import letterbox
from scripts.rknn_postprocess import dfl, post_process
from scripts.run_demo import run


def analyzer(**kwargs):
    return ProximityAnalyzer('', {0:'forklift',1:'person'}, model=object(),
                             **kwargs)


def test_driver_filter_keeps_standing_pedestrian():
    assert _is_aboard([20,10,40,40], [0,0,100,100])
    assert not _is_aboard([20,10,40,100], [0,0,100,100])


def test_dfl_matches_torch():
    import torch
    x = np.random.default_rng(1).normal(size=(1,64,3,3)).astype(np.float32)
    ref = (torch.tensor(x).reshape(1,4,16,3,3).softmax(2) *
           torch.arange(16).reshape(1,1,16,1,1)).sum(2).numpy()
    np.testing.assert_allclose(dfl(x),ref,rtol=1e-6)


def test_letterbox_mapping_and_empty_detections():
    frame = np.zeros((721,1280,3),np.uint8)
    image,(rx,ry,left,top) = letterbox(frame)
    assert image.shape == (640,640,3)
    assert rx*1280+2*left == 640
    assert abs(ry*721+2*top-640) <= 1
    outputs = []
    for size in (80,40,20):
        outputs += [np.zeros((1,c,size,size),np.float32) for c in (64,2,1)]
    assert post_process(outputs) == (None,None,None)


def test_closing_speed_uses_elapsed_time_and_flush(tmp_path):
    a = analyzer()
    a.frame_idx, a._fps_hint, a._time_s = 1,24,0.0
    a._update_track(1,'forklift',[0,0,10,10])
    a._update_track(2,'person',[200,0,210,10])
    a._update_pairs({1},{2})
    a.frame_idx, a._time_s = 2,2.0
    a._update_track(2,'person',[100,0,110,10])
    a._update_pairs({1},{2})
    pair = a.pairs[1,2]
    assert pair.smoothed_closing_speed_mps == pytest.approx(.5)
    assert pair.tier == 'CRITICAL'
    a.flush_open_events()
    a.flush_open_events()
    assert len(a.log_rows) == 1
    assert a.log_rows[0]['duration_s'] == 2.0
    assert pair.closing_speed_history.maxlen == 5
    a.save_log(tmp_path/'events.csv')
    assert 'CRITICAL' in (tmp_path/'events.csv').read_text()


def test_smoothing_config_and_empty_log(tmp_path):
    a = analyzer(closing_speed_smoothing_window=9)
    a.frame_idx,a._time_s,a._fps_hint = 1,1.,24
    a._update_track(1,'forklift',[0,0,10,10])
    a._update_track(2,'person',[500,0,510,10])
    a._update_pairs({1},{2})
    assert a.pairs[1,2].closing_speed_history.maxlen == 9
    a.save_log(tmp_path/'empty.csv')
    assert len((tmp_path/'empty.csv').read_text().splitlines()) == 1


def test_bad_source_fails(tmp_path):
    with pytest.raises(RuntimeError, match='Cannot open'):
        run(str(tmp_path/'missing.mp4'),'missing.rknn',output=tmp_path/'out')


def test_nearmiss_person_is_critical_at_5_and_8_seconds():
    # Measured boxes from the original sample; center-distance mode misses both.
    for forklift,person in [
        ([385.33,193.79,583.65,552.63],[773.83,249.65,918.95,617.66]),
        ([303.69,193.71,596.94,560.19],[892.93,254.04,1001.12,521.20]),
    ]:
        assert not _is_aboard(person,forklift)
        ratio,_ = _image_zone_gap(person,forklift)
        assert ratio == 0
        a=analyzer(risk_mode='image_zone')
        a.frame_idx,a._time_s,a._fps_hint=1,1.,24
        a._update_track(1,'forklift',forklift)
        a._update_track(2,'person',person)
        a._update_pairs({1},{2})
        assert a.pairs[1,2].tier == 'CRITICAL'
        a.flush_open_events()
        assert a.log_rows[0]['min_distance_m'] is None


def test_zone_progression_and_critical_clear_hysteresis():
    # Forklift ellipse: center (100,200), radii (480,80).
    a=analyzer(risk_mode='image_zone')
    a._fps_hint=24
    for frame,radius,expected in [(1,1.3,'SAFE'),(2,1.0,'CAUTION'),
                                  (3,.8,'CRITICAL'),(4,.90,'CRITICAL'),
                                  (5,.96,'CAUTION'),(6,1.3,'SAFE')]:
        a.frame_idx,a._time_s=frame,frame/24
        a._update_track(1,'forklift',[0,0,200,200])
        x=100+480*radius
        a._update_track(2,'person',[x,100,x+40,200])
        a._update_pairs({1},{2})
        assert a.pairs[1,2].tier == expected
