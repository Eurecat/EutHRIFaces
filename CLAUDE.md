# EutHRIFaces — agent guide

Face perception for HRI: detection + 5 landmarks, persistent face identity (`U<n>`, MongoDB),
gaze, lip-based speaking. Component of EutPerceptionStack (stack-wide guide: `../../CLAUDE.md`
when cloned under `stack/`). `AGENTS.md` is a symlink to this file.

## Layout

```
face_detection/          YOLO face (ONNX) + landmarks, optional BOXMOT tracking
face_recognition/        embeddings + identity_manager.py (U<n> lifecycle, Mongo gallery); tools/ = offline eval
gaze_estimation/         head pose + gaze
visual_speech_activity/  lip movement → speaking
Docker/                  Dockerfile(.arm), build_container.sh, docker-compose.yaml, tiago-eth-docker-compose.yaml,
                         deps.repos → deps/hri_msgs (jazzy-devel), .env.example, entrypoint.sh
docs/multi_robot.md      shared face gallery: scope, provenance, refresh, tombstones
plans/                   dev notes (face_identity_experiments.md: every identity experiment + numbers)
```

Each package: `launch/<pkg>.launch.py`, `config/<pkg>_params.yaml`, `test/`.

## Topics (all relative; prefixed by the robot namespace in multi-robot)

| Node | In | Out |
|---|---|---|
| face_detection | `camera/image_raw/compressed` | `humans/faces/detected` (`FacialLandmarksArray`; empty array = no face in that frame) |
| face_recognition | image + detections | `humans/faces/recognized` (`recognized_face_id`, `identity_status`, `face_quality`) |
| gaze_estimation | detections | `humans/faces/gaze` |
| visual_speech_activity | detections | `humans/faces/speaking` |

`ros4hri_with_id:=true` switches to per-id REP-155 topics (`humans/faces/<id>/…`).

## Build / run / test

```bash
cd Docker && ./build_container.sh [--arm|--cpu|--humble|--vulcanexus]   # image eut_human_face[_arm|_cpu|_vulcanexus]:<distro>
docker compose up -d        # 4 nodes + mongodb_faces_service (27018) + mongo_express_faces (8082)
# inside a container (/workspace colcon ws, venv /opt/ros_python_env):
colcon test --packages-select face_detection face_recognition gaze_estimation visual_speech_activity
cd face_recognition && python3 -m pytest test/test_identity_manager.py      # identity rules, no ROS
```

Offline identity evaluation (record → label → replay, no GPU): README "Evaluating identity changes offline".

## Identity rules (face_recognition/identity_manager.py)

TENTATIVE → CONFIRMED after `min_confirm_samples` over `min_confirm_seconds`; only confirmed ids
are long-term keys. Low-quality faces may match but never create/teach. Fragments merge except when
seen in the same frame. `U` numbers are never reused. Gallery documents are keyed by `model_key`
(profile scope); documents without it are ignored.

Multi-robot env: `FACE_DB_MONGO_URI`, `FACE_PROFILE_SCOPE` (must match on all robots),
`FACE_GALLERY_REFRESH_S` (15), `ROBOT_ID`. Refresh runs on its own timer, never in the frame path.

## Traps

- Params YAML is passed as a dict of its `ros__parameters`, so a namespaced node keeps its config.
  Keep that pattern in new launch files; verify with `ros2 param get`, not topic names.
- `DOCKER_RUNTIME` must be `nvidia`: under runc ONNX CUDA segfaults (face_detector exits -11).
- MediaPipe `face_landmarker.task` is restored from `/opt/mediapipe_weights/` by the entrypoint
  when the mounted copy is missing or truncated.
- Weights are auto-downloaded into `<pkg>/weights/` (gitignored).
- `main` is protected: changes reach it through a pull request.

## Commits

Plain human sentences describing the change, no prefixes. No AI attribution: no
`Co-Authored-By` trailer, no "Generated with" line.
