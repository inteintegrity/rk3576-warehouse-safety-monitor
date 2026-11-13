#!/usr/bin/env bash
# Final demo launcher. Run from any directory; no sudo or system-library changes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ACTION="${1:-help}"
if (( $# )); then shift; fi

help() {
  cat <<'HELP'
Usage: bash run.sh <command> [options]
  setup       Create .venv, install board dependencies, and fetch the sample
  web         Start the NPU dashboard on port 8088 (Ctrl+C stops it)
  video       Analyze the sample with the NPU; outputs go to outputs/demo
  render      Render a glowing showcase from a run directory (default outputs/demo)
  check       Check dependencies and the included RKNN model hashes
  test        Install test dependencies and run regression checks

Examples:
  bash run.sh setup
  bash run.sh web
  bash run.sh web --port 8089
  bash run.sh video --source /path/to/video.mp4
  bash run.sh render outputs/demo

Optional: WAREHOUSE_PYTHON=/path/to/existing/environment/bin/python
HELP
}
case "$ACTION" in help|-h|--help) help; exit 0;; esac
case "$ACTION" in setup|web|video|render|check|test) ;; *) help >&2; exit 2;; esac

if [[ -n "${WAREHOUSE_PYTHON:-}" ]]; then
  PYTHON="$WAREHOUSE_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ "$ACTION" == setup ]]; then
  python3 -c 'import sys; assert sys.version_info[:2] == (3,11), "Board setup requires Python 3.11"'
  python3 -m venv "$ROOT/.venv"
  PYTHON="$ROOT/.venv/bin/python"
else
  echo 'Run bash run.sh setup first, or set WAREHOUSE_PYTHON to an existing environment.' >&2
  exit 1
fi

case "$ACTION" in
  setup)
    "$PYTHON" -c 'import platform,sys; assert platform.system()=="Linux" and platform.machine()=="aarch64", "Board setup requires ARM64 Linux"; assert sys.version_info[:2]==(3,11), "Board setup requires Python 3.11"'
    if ! "$PYTHON" -c 'import torch; from importlib.metadata import version; version("torchvision")' >/dev/null 2>&1; then
      "$PYTHON" -m pip install torch==2.2.0 torchvision==0.17.0
    fi
    "$PYTHON" -m pip install -r requirements.txt
    "$PYTHON" -m pip install rknn-toolkit-lite2==2.3.2
    "$PYTHON" tools/fetch_sample.py
    "$PYTHON" tools/check_runtime.py
    echo 'Ready. Start the dashboard with: bash run.sh web'
    ;;
  web)
    "$PYTHON" tools/fetch_sample.py
    exec "$PYTHON" server.py --backend rknn --host 0.0.0.0 --port 8088 "$@"
    ;;
  video)
    "$PYTHON" tools/fetch_sample.py
    exec "$PYTHON" scripts/run_demo.py --backend rknn --model models/warehouse_yolo11_rk3576_int8.rknn --output outputs/demo "$@"
    ;;
  render)
    command -v ffmpeg >/dev/null || { echo 'Install FFmpeg on the rendering machine first.' >&2; exit 1; }
    "$PYTHON" tools/fetch_sample.py
    "$PYTHON" tools/render_halo.py --run "${1:-outputs/demo}" --output outputs/showcase/warehouse-safety-master.mp4
    exec ffmpeg -y -hide_banner -loglevel warning -i outputs/showcase/warehouse-safety-master.mp4 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart outputs/showcase/warehouse-safety.mp4
    ;;
  check) exec "$PYTHON" tools/check_runtime.py ;;
  test)
    "$PYTHON" -m pip install -r requirements-dev.txt
    exec "$PYTHON" -m pytest tests -q --basetemp .test-tmp "$@"
    ;;
esac
