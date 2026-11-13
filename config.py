"""Final RK3576 warehouse demo defaults."""
MODEL_WEIGHTS = "models/warehouse_yolo11_rk3576_int8.rknn"
WAREHOUSE_CLASSES = {0: "forklift", 1: "person"}
CONFIDENCE_THRESHOLD = 0.35
DEFAULT_INPUT_VIDEO = "data/test_clip_nearmiss.mp4"
