#!/usr/bin/env bash
# Extract MediaPipe's face landmark network from face_landmarker.task and convert it to ONNX
# for landmark_backend "facemesh_onnx" (face_detection/weights/face_landmarks_detector.onnx).
#
# Needs internet; TensorFlow + tf2onnx go into a throwaway virtualenv (~1 GB, a few minutes),
# nothing is installed into the container image or the system Python.
#
# Usage: tools/convert_face_landmarks_to_onnx.sh [face_landmarker.task] [output.onnx]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-$HERE/weights/face_landmarker.task}"
OUT="${2:-$HERE/weights/face_landmarks_detector.onnx}"
TASK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ ! -f "$TASK" ]; then
    echo "Downloading $TASK_URL"
    curl -fL "$TASK_URL" -o "$WORK/face_landmarker.task"
    TASK="$WORK/face_landmarker.task"
fi

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q tensorflow tf2onnx
"$WORK/venv/bin/python" - "$TASK" "$WORK" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extract("face_landmarks_detector.tflite", sys.argv[2])
PY
"$WORK/venv/bin/python" -m tf2onnx.convert --tflite "$WORK/face_landmarks_detector.tflite" \
    --output "$OUT" --opset 17
echo "Wrote $OUT"
