# GPT README: EutHRIFaces

ROS2 packages for face-related perception capabilities in Human-Robot Interaction (HRI) applications.

This repository contains three main packages:

## Packages

### 1. face_detection 🔍
**Status**: ✅ **Implemented**

YOLO-based face detection with 5 key facial landmarks.

- **Input**: RGB camera images
- **Output**: `hri_msgs/FacialLandmarks` (ros4hri compatible)
- **Features**: 
  - Auto-download YOLO face model
  - CPU/GPU support
  - Real-time performance
  - 5 key landmarks (eyes, nose, mouth corners)

[➡️ See face_detection README](face_detection/README.md)

### 2. face_recognition 👤
**Status**: 🚧 **TODO - Not Implemented**

Face recognition and identification capabilities.

- **Planned Features**:
  - Face embedding extraction
  - Identity management
  - Face matching and verification
  - Database integration

### 3. gaze_estimation 👁️
**Status**: 🚧 **TODO - Not Implemented**

Gaze direction estimation from facial landmarks.

- **Planned Features**:
  - Head pose estimation
  - Eye gaze direction
  - Point of attention estimation
  - 3D gaze vectors

## Quick Start

### Prerequisites
- ROS2 (Humble/Iron)
- Python 3.8+
- OpenCV
- ONNXRuntime

### Installation

1. Clone into your ROS2 workspace:
```bash
cd your_ws/src
git clone <repository_url>
```

2. Install dependencies:
```bash
cd your_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build packages:
```bash
colcon build --packages-select face_detection face_recognition gaze_estimation
source install/setup.bash
```

### Usage

#### Face Detection
```bash
# Launch face detection node
ros2 launch face_detection face_detection.launch.py

# With custom camera topic
ros2 launch face_detection face_detection.launch.py input_topic:=/your/camera/topic
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
