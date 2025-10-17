# README: EutHRIFaces

ROS2 packages for face-related perception capabilities in Human-Robot Interaction (HRI) applications.

This repository contains three main packages:

## Packages

### 1. face_detection 🔍

YOLO-based face detection with 5 key facial landmarks.

- **Input**: RGB camera images
- **Output**: `hri_msgs/FacialLandmarksArray` (ros4hri compatible)
- **Features**: 
  - Auto-download YOLO face model
  - CPU/GPU support
  - Real-time performance
  - 5 key landmarks (eyes, nose, mouth corners)

[➡️ See face_detection README](face_detection/README.md)

### 2. face_recognition 👤
Face recognition and identification capabilities.

- **Planned Features**:
  - Face embedding extraction
  - Identity management
  - Face matching and verification
  - Database integration

### 3. gaze_estimation 👁️
Gaze direction estimation from facial landmarks.

- **Planned Features**:
  - Head pose estimation
  - Eye gaze direction
  - Point of attention estimation
  - 3D gaze vectors

## Quick Start

## Installation & Setup

#### 0. Build Base Image

First, build the desired base Docker image from [EutRobAIDockers](https://github.com/Eurecat/EutRobAIDockers)

#### 1. Clone This Repository

```bash
git clone git@github.com:Eurecat/EutHRIFaces.git
cd EutHRIFaces
```

#### 2. Build the application image

   ```bash
   cd Docker && ./build_container.sh --vulcanexus
   ```
   Please note that:
    * your default ssh keys will be used to build the image
    * you might need to be within Eurecat VPN to pull dependencies from our private gitlab through vcs. 
   
   You can use `--clean-rebuild` to force a clean rebuild from scratch (i.e. no cached layers).

## Launch

### Option A: Deployment

As simple as...
   ```bash
   docker compose up
   ```
... within `Docker/` folder

Will start all face processing modules (face detection, face recognition, and gaze estimation).

If you want to run only specific modules, you can scale down the services you don't need:

   ```bash
   # Run only face detection
   docker compose up --scale eut_face_detection=0
   
   # Run only face detection and gaze estimation
   docker compose up --scale eut_face_detection=0 --scale eut_gaze_estimation=0
   
   # Run only face detection and face recognition
   docker compose up --scale eut_face_detection=0 --scale eut_face_recognition=0

   ```

### Option B: DevContainer (Development)

Within VS Code editor, make sure you have installed extension DevContainer, press `ctrl+shit+P` (command option) and search for "_Dev Containers: Open Folder in Container..._". From there you can select the folder Docker/DevContainer and the stack will launch in development mode (no node will be automatically started).

### Notes
Please note that launching the stack might involve launch of GUI application from docker, therefore make sure in the current active session in the host you have given at least once the following command to make sure permissions are given.

```bash
xhost +local:docker
```

### Usage
In terminal inside the docker:

#### Face Detection
```bash
# Launch face detection node
ros2 launch face_detection face_detection.launch.py

# With custom camera topic
ros2 launch face_detection face_detection.launch.py input_topic:=/your/camera/topic
```

#### Face Recognition
```bash
# Launch face recognition node
ros2 launch face_recognition face_recognition.launch.py
```

#### Gaze Estimation
```bash
# Launch gaze estimation node
ros2 launch gaze_estimation gaze_estimation.launch.py
```

## Docker Support

The repository includes Docker support in the `Docker/` directory for easy deployment and development.

## ROS4HRI Compatibility

All packages follow the [ros4hri](https://github.com/ros4hri) standard for human perception in robotics:

- Uses standard `hri_msgs` message definitions
- Compatible with other ros4hri packages
- Follows established conventions for human tracking and identification

## Architecture

```
EutHRIFaces/
├── face_detection/     # YOLO face detection (IMPLEMENTED)
├── face_recognition/   # Face identification (TODO)
├── gaze_estimation/    # Gaze direction (TODO)
└── Docker/            # Docker deployment files
```

## Dependencies

- **hri_msgs**: ROS4HRI message definitions
- **cv_bridge**: OpenCV-ROS bridge
- **sensor_msgs**: Standard sensor messages
- **std_msgs**: Standard message types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Apache-2.0

## Maintainer

Josep Bravo (josep.bravo@eurecat.org)

## Related Packages

- [EutHRIHumanBody](../EutHRIHumanBody): Skeleton and pose detection
- [EutEntityDetection](../EutEntityDetection): General object detection
- [eut_yolo](../eut_yolo): Advanced YOLO-based perception pipeline
