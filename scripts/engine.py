"""
Core analytics engine: wraps a YOLO model + ByteTrack tracker to produce,
frame by frame:
  - bounding boxes with persistent track IDs for person and forklift
  - live distance + closing speed for every forklift<->person pair
  - a graduated SAFE / CAUTION / CRITICAL alert per pair
  - an annotated frame ready to display or write to video
  - a CSV-ready log of proximity events (one row per continuous span a pair
    spends at CAUTION or above)

Designed to be imported both by the CLI script (detect_proximity.py) and
the Streamlit app (app.py), so the logic only lives in one place.

The release uses the trained warehouse detector for both classes. The earlier
color-blob stand-in is omitted from this final deployment package.
"""
from collections import deque
from dataclasses import dataclass, field
import csv
import time

import cv2
import numpy as np


SAFE, CAUTION, CRITICAL = "SAFE", "CAUTION", "CRITICAL"
_TIER_RANK = {SAFE: 0, CAUTION: 1, CRITICAL: 2}
_TIER_COLOR = {SAFE: (60, 200, 60), CAUTION: (0, 200, 255), CRITICAL: (0, 0, 255)}
IMAGE_CRITICAL_RADIUS = 0.85
IMAGE_CAUTION_MARGIN = 0.20
IMAGE_CRITICAL_HYSTERESIS = 0.08



@dataclass
class TrackState:
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # ground-contact pts
    last_seen_frame: int = 0
    box: tuple = None


@dataclass
class PairState:
    forklift_id: int
    person_id: int
    last_distance_m: float = None
    closing_speed_history: deque = field(default_factory=lambda: deque(maxlen=5))
    smoothed_closing_speed_mps: float = 0.0
    last_seen_frame: int = 0
    tier: str = SAFE
    event_open: bool = False
    event_start_frame: int = None
    event_min_distance_m: float = None
    event_peak_tier: str = None
    last_seen_s: float = None
    event_start_s: float = None
    image_gap_ratio: float = None
    alert_reason: str = 'legacy_distance_and_closing_speed'


def _ground_contact_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def _vehicle_zone(box, extension=1.0):
    """Conservative IMAGE region; does not estimate a physical vehicle footprint.

    Extend both sides because a single detector box does not reveal heading,
    forks or load dimensions. Values must be checked for each camera.
    """
    x1,y1,x2,y2 = [float(x) for x in box]
    w,h = x2-x1,y2-y1
    return (x1-extension*w, y2-.25*h, x2+extension*w, y2+.20*h)


def _image_zone_gap(person_box, forklift_box, extension=1.0):
    cx,cy,rx,ry = _vehicle_ellipse(forklift_box,extension)
    # Use the closest point of the pedestrian's bottom edge, rather than
    # their center. A person's near side can enter a zone before their center.
    px = min(max(cx,float(person_box[0])),float(person_box[2]))
    py = float(person_box[3])
    dx,dy = (float(px)-cx)/rx,(float(py)-cy)/ry
    radius = float(np.hypot(dx,dy))
    scale = min(1.,IMAGE_CRITICAL_RADIUS/max(radius,1e-6))
    nearest = (cx+(float(px)-cx)*scale,cy+(float(py)-cy)*scale)
    return max(radius-IMAGE_CRITICAL_RADIUS,0.), nearest


def _vehicle_ellipse(box,extension=1.0):
    """Image-space ellipse only; perspective/physical radius is not calibrated."""
    x1,y1,x2,y2 = [float(x) for x in box]
    return ((x1+x2)/2,y2,max((.5+1.9*extension)*(x2-x1),1.),max(.4*(y2-y1),1.))


# A rider's box bottom (visible torso/lap, since the cab hides their legs)
# sits well above the forklift's own ground-contact line, unlike a standing
# pedestrian whose feet reach the floor at roughly the same image row as the
# forklift's wheels. Horizontal overlap alone isn't a safe signal -- a
# pedestrian standing right beside the forklift (the exact near-miss case we
# want to keep) can overlap it horizontally too.
#
# Normalized against the PERSON's own box height, not the forklift's: the
# forklift's box height swings a lot depending on how much of the mast is
# visible/detected (a short mock chassis box vs. a full mast+chassis box
# from real fine-tuned weights), which silently shifts what counts as
# "aboard" if used as the denominator -- this bit a real fine-tuned model
# whose taller forklift boxes dropped a genuine driver's ratio from 0.36
# (correctly filtered against the old mock-forklift-calibrated threshold) to
# 0.31 (incorrectly not filtered). A person's own box height is stable
# regardless of which forklift-detection method produced the paired box.
ABOARD_HEIGHT_RATIO = 0.4


def _is_aboard(person_box, forklift_box):
    """True if `person_box` looks like it's riding `forklift_box` (operator), not standing beside it."""
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = forklift_box
    horizontal_overlap = min(px2, fx2) - max(px1, fx1)
    if horizontal_overlap <= 0:
        return False
    person_height = max(py2 - py1, 1e-6)
    elevation = fy2 - py2  # how far above the forklift's ground line the person's box bottom sits
    return (elevation / person_height) >= ABOARD_HEIGHT_RATIO


class ProximityAnalyzer:
    def __init__(
        self,
        weights_path,
        warehouse_classes,
        confidence=0.35,
        tracker_config="bytetrack.yaml",
        caution_distance_m=3.0,
        critical_distance_m=1.5,
        closing_speed_caution_mps=1.0,
        closing_speed_smoothing_window=5,
        pair_stale_frames=15,
        meters_per_pixel=0.01,
        model=None,
        risk_mode='legacy',
        vehicle_extension=1.0,
        image_caution_ratio=IMAGE_CAUTION_MARGIN,
    ):
        # RK3576 adapter supplies the same track() result contract as YOLO.
        if model is None:
            from ultralytics import YOLO
            model = YOLO(weights_path)
        self.model = model
        if risk_mode not in ('legacy', 'image_zone'):
            raise ValueError(risk_mode)
        self.risk_mode = risk_mode
        self.vehicle_extension = vehicle_extension
        self.image_caution_ratio = image_caution_ratio
        self.warehouse_classes = warehouse_classes  # {class_id: class_name}
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.caution_distance_m = caution_distance_m
        self.critical_distance_m = critical_distance_m
        self.closing_speed_caution_mps = closing_speed_caution_mps
        self.closing_speed_smoothing_window = closing_speed_smoothing_window
        self.pair_stale_frames = pair_stale_frames
        self.meters_per_pixel = meters_per_pixel

        self.tracks: dict[int, TrackState] = {}
        self.pairs: dict[tuple[int, int], PairState] = {}
        self.log_rows = []
        self.frame_idx = 0

    # ------------------------------------------------------------------
    def process_frame(self, frame, fps, timestamp_s=None):
        """Run detection+tracking on one BGR frame. Returns annotated frame."""
        self.frame_idx += 1
        self._fps_hint = fps
        self._time_s = self.frame_idx / fps if timestamp_s is None else timestamp_s
        annotated = frame.copy()

        forklift_boxes = {}  # track_id -> box, this frame only
        person_boxes = {}

        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            tracker=self.tracker_config,
            classes=list(self.warehouse_classes.keys()),
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                track_id, cls_id = int(track_id), int(cls_id)
                class_name = self.warehouse_classes.get(cls_id, str(cls_id))
                self._update_track(track_id, class_name, box)
                if class_name == "forklift":
                    forklift_boxes[track_id] = box
                elif class_name == "person":
                    person_boxes[track_id] = box
                self._draw_box(annotated, box, track_id, class_name)

        # Exclude the forklift's own operator (a "person" box sitting on/in
        # the vehicle) from proximity pairing -- they aren't a separate
        # at-risk pedestrian, and pairing them with their own forklift would
        # permanently read as near-zero distance / CRITICAL.
        at_risk_person_ids = {
            p_id for p_id, p_box in person_boxes.items()
            if not any(_is_aboard(p_box, f_box) for f_box in forklift_boxes.values())
        }
        self.last_observation = {
            'frame': self.frame_idx,
            'forklifts': {int(k): [float(x) for x in v] for k,v in forklift_boxes.items()},
            'people': {int(k): {'box': [float(x) for x in v],
                              'excluded_as_operator': k not in at_risk_person_ids}
                       for k,v in person_boxes.items()},
        }

        forklift_ids_seen = set(forklift_boxes.keys())
        self._update_pairs(forklift_ids_seen, at_risk_person_ids)
        self._gc_stale_pairs(forklift_ids_seen, at_risk_person_ids)
        self._draw_pairs(annotated, forklift_ids_seen, at_risk_person_ids)
        if self.risk_mode == 'image_zone':
            for box in forklift_boxes.values():
                cx,cy,rx,ry = _vehicle_ellipse(box,self.vehicle_extension)
                for radius,color in ((1.35,(80,220,100)),(IMAGE_CRITICAL_RADIUS+self.image_caution_ratio,(0,200,255)),(IMAGE_CRITICAL_RADIUS,(60,80,255))):
                    cv2.ellipse(annotated,(int(cx),int(cy)),(int(rx*radius),int(ry*radius)),0,0,360,color,1)
        self._draw_hud(annotated, forklift_ids_seen, at_risk_person_ids)
        return annotated

    # ------------------------------------------------------------------
    def _update_track(self, track_id, class_name, box):
        state = self.tracks.setdefault(track_id, TrackState(class_name=class_name))
        state.positions.append(_ground_contact_point(box))
        state.last_seen_frame = self.frame_idx
        state.box = tuple(box)

    def _update_pairs(self, forklift_ids_seen, person_ids_seen):
        for f_id in forklift_ids_seen:
            for p_id in person_ids_seen:
                key = (f_id, p_id)
                if key not in self.pairs:
                    self.pairs[key] = PairState(
                        f_id, p_id, closing_speed_history=deque(maxlen=self.closing_speed_smoothing_window)
                    )
                pair = self.pairs[key]
                self._update_pair(pair)

    def _update_pair(self, pair):
        f_point = self.tracks[pair.forklift_id].positions[-1]
        p_point = self.tracks[pair.person_id].positions[-1]
        dist_px = np.hypot(f_point[0] - p_point[0], f_point[1] - p_point[1])
        dist_m = float(dist_px * self.meters_per_pixel)

        if pair.last_distance_m is not None:
            dt = max(self._time_s - pair.last_seen_s, 1e-6)
            closing_mps = (pair.last_distance_m - dist_m) / dt
            pair.closing_speed_history.append(closing_mps)
            pair.smoothed_closing_speed_mps = float(np.mean(pair.closing_speed_history))

        pair.last_distance_m = dist_m
        pair.last_seen_frame = self.frame_idx
        pair.last_seen_s = self._time_s

        if self.risk_mode == 'image_zone':
            ratio,_ = _image_zone_gap(self.tracks[pair.person_id].box,
                                      self.tracks[pair.forklift_id].box,self.vehicle_extension)
            pair.image_gap_ratio = ratio
            # Hysteresis: clear CAUTION only after leaving a slightly wider band.
            caution = self.image_caution_ratio + (0.15 if pair.tier != SAFE else 0.)
            critical = IMAGE_CRITICAL_HYSTERESIS if pair.tier == CRITICAL else 0.
            tier = CRITICAL if ratio <= critical else CAUTION if ratio <= caution else SAFE
            pair.alert_reason = ('projected_zone_entry' if tier == CRITICAL else
                                 'near_projected_zone' if tier == CAUTION else 'outside_warning_zone')
        else:
            tier = self._classify_tier(dist_m, pair.smoothed_closing_speed_mps)
        self._apply_tier(pair, tier, dist_m)

    def _classify_tier(self, dist_m, closing_speed_mps):
        if dist_m <= self.critical_distance_m and closing_speed_mps > 0:
            return CRITICAL
        if dist_m <= self.caution_distance_m or closing_speed_mps >= self.closing_speed_caution_mps:
            return CAUTION
        return SAFE

    def _apply_tier(self, pair, tier, dist_m):
        if tier != SAFE and not pair.event_open:
            pair.event_open = True
            pair.event_start_frame = self.frame_idx
            pair.event_start_s = self._time_s
            pair.event_min_distance_m = dist_m
            pair.event_peak_tier = tier
        elif pair.event_open:
            pair.event_min_distance_m = min(pair.event_min_distance_m, dist_m)
            if _TIER_RANK[tier] > _TIER_RANK[pair.event_peak_tier]:
                pair.event_peak_tier = tier
            if tier == SAFE:
                self._finalize_event(pair, end_frame=self.frame_idx)
        pair.tier = tier

    def _finalize_event(self, pair, end_frame):
        fps = self._fps_hint
        end_s = self._time_s if end_frame == self.frame_idx else pair.last_seen_s
        row = {
                "forklift_id": pair.forklift_id,
                "person_id": pair.person_id,
                "start_frame": pair.event_start_frame,
                "start_timestamp_s": round(pair.event_start_s, 2),
                "end_frame": end_frame,
                "end_timestamp_s": round(end_s, 2),
                "duration_s": round(end_s - pair.event_start_s, 2),
                "min_distance_m": round(pair.event_min_distance_m, 2),
                "peak_tier": pair.event_peak_tier,
            }
        if self.risk_mode == 'image_zone':
            row['risk_mode'] = 'image_zone'
            row['alert_basis'] = 'projected_vehicle_zone; no metric distance claim'
            # Keep legacy CSV shape, but do not present an uncalibrated estimate as meters.
            row['min_distance_m'] = None
        self.log_rows.append(row)
        pair.event_open = False
        pair.event_start_frame = None
        pair.event_min_distance_m = None
        pair.event_peak_tier = None

    def _gc_stale_pairs(self, forklift_ids_seen, person_ids_seen):
        present = {(f, p) for f in forklift_ids_seen for p in person_ids_seen}
        for key, pair in list(self.pairs.items()):
            if key in present:
                continue
            if self.frame_idx - pair.last_seen_frame > self.pair_stale_frames:
                if pair.event_open:
                    self._finalize_event(pair, end_frame=pair.last_seen_frame)
                del self.pairs[key]
        for key, track in list(self.tracks.items()):
            if self.frame_idx - track.last_seen_frame > self.pair_stale_frames:
                del self.tracks[key]

    def flush_open_events(self):
        """Finalize any still-open events at end of video, at their last-seen frame."""
        for pair in list(self.pairs.values()):
            if pair.event_open:
                self._finalize_event(pair, end_frame=pair.last_seen_frame)

    # ------------------------------------------------------------------
    def _draw_box(self, frame, box, track_id, class_name, is_mock=False):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 200, 255) if class_name == "forklift" else (255, 140, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{'~' if is_mock else ''}{class_name} #{track_id}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_pairs(self, frame, forklift_ids_seen, person_ids_seen):
        for f_id in forklift_ids_seen:
            for p_id in person_ids_seen:
                pair = self.pairs.get((f_id, p_id))
                if pair is None or pair.tier == SAFE:
                    continue
                f_point = tuple(int(v) for v in self.tracks[f_id].positions[-1])
                p_point = tuple(int(v) for v in self.tracks[p_id].positions[-1])
                if self.risk_mode == 'image_zone':
                    _,nearest = _image_zone_gap(self.tracks[p_id].box,self.tracks[f_id].box,self.vehicle_extension)
                    f_point = tuple(int(v) for v in nearest)
                color = _TIER_COLOR[pair.tier]
                if self.risk_mode == 'image_zone':
                    x1,y1,x2,y2 = [int(v) for v in self.tracks[p_id].box]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                    label = 'DANGER' if pair.tier == CRITICAL else 'CAUTION'
                    cv2.putText(frame,f'PERSON #{p_id} | {label}',(x1,max(y1-8,15)),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
                cv2.line(frame, f_point, p_point, color, 2)
                mid = ((f_point[0] + p_point[0]) // 2, (f_point[1] + p_point[1]) // 2)
                cv2.putText(
                    frame, (f"{pair.tier} | VEHICLE ZONE" if self.risk_mode == 'image_zone'
                            else f"{pair.tier} {pair.last_distance_m:.1f}m"), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

    def _draw_hud(self, frame, forklift_ids_seen, person_ids_seen):
        tier_counts = {SAFE: 0, CAUTION: 0, CRITICAL: 0}
        for f_id in forklift_ids_seen:
            for p_id in person_ids_seen:
                pair = self.pairs.get((f_id, p_id))
                if pair is not None:
                    tier_counts[pair.tier] += 1

        lines = [
            (f"Forklifts: {len(forklift_ids_seen)}  Pedestrians: {len(person_ids_seen)}", (255, 255, 255)),
            (f"CAUTION pairs: {tier_counts[CAUTION]}", _TIER_COLOR[CAUTION] if tier_counts[CAUTION] else (255, 255, 255)),
            (f"CRITICAL pairs: {tier_counts[CRITICAL]}", _TIER_COLOR[CRITICAL] if tier_counts[CRITICAL] else (255, 255, 255)),
        ]
        y = 28
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 20 + 22 * len(lines)), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for line, color in lines:
            cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 22

    # ------------------------------------------------------------------
    def save_log(self, csv_path):
        keys = ['forklift_id', 'person_id', 'start_frame', 'start_timestamp_s',
                'end_frame', 'end_timestamp_s', 'duration_s', 'min_distance_m', 'peak_tier']
        if self.risk_mode == 'image_zone': keys += ['risk_mode','alert_basis']
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_rows)


def run_on_video(analyzer: ProximityAnalyzer, input_path, output_path=None, csv_path=None, max_frames=None):
    """Convenience runner used by both the CLI script and quick tests."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = analyzer.process_frame(frame, fps)
        if writer:
            writer.write(annotated)
        n += 1
        if max_frames and n >= max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    analyzer.flush_open_events()
    if csv_path:
        analyzer.save_log(csv_path)

    elapsed = time.time() - start
    return {
        "frames_processed": n,
        "elapsed_s": round(elapsed, 2),
        "fps_processed": round(n / elapsed, 1) if elapsed > 0 else 0,
        "events_logged": len(analyzer.log_rows),
    }
