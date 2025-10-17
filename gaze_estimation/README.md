# Gaze Estimation Package

This ROS2 package provides gaze estimation capabilities for Human-Robot Interaction (HRI) applications. It subscribes to facial landmarks from face detection and computes gaze direction and confidence scores using a pinhole camera model.

## Overview

The gaze estimation node:
- Subscribes to `FacialLandmarksArray` messages from face detection
- Extracts 5 key facial landmarks (eyes, nose, mouth corners)
- Uses OpenCV's solvePnP algorithm with a 3D face model to estimate head pose
- Computes gaze direction vector and confidence score
- Publishes `Gaze` messages following the ros4hri standard

## Features

- **Pinhole Camera Model**: Configurable camera intrinsic parameters
- **Real-time Processing**: Efficient computation suitable for real-time applications
- **ROS4HRI Compliance**: Uses standard HRI message types
- **Configurable Parameters**: Tunable via YAML config files and launch parameters
- **Debug Output**: Optional debug logging for development

## Dependencies

- ROS2 (Humble or later)
- Python 3.8+
- OpenCV (python3-opencv)
- NumPy (python3-numpy)
- hri_msgs package

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
# Package should already be in EutHRIFaces
```

2. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the package:
```bash
colcon build --packages-select gaze_estimation
```

4. Source the workspace:
```bash
source install/setup.bash
```

## Usage

### Basic Usage

Launch the gaze estimation node:
```bash
ros2 launch gaze_estimation gaze_estimation.launch.py
```

### With Custom Parameters

Launch with custom camera parameters:
```bash
ros2 launch gaze_estimation gaze_estimation.launch.py \
    focal_length:=800.0 \
    image_width:=1280 \
    image_height:=720 \
    center_x:=640.0 \
    center_y:=360.0
```

### With Custom Topics

Launch with custom input/output topics:
```bash
ros2 launch gaze_estimation gaze_estimation.launch.py \
    input_topic:=/humans/faces/detected \
    output_topic:=/humans/faces/gaze
```

## Configuration

### Parameters

The node accepts the following parameters:

#### Topics
- `input_topic` (string): Input topic for FacialLandmarksArray messages
  - Default: `/humans/faces/detected`
- `output_topic` (string): Output topic for Gaze messages  
  - Default: `/humans/faces/gaze`

#### Camera Parameters
- `image_width` (int): Image width in pixels
  - Default: `640`
- `image_height` (int): Image height in pixels
  - Default: `480`
- `focal_length` (double): Camera focal length
  - Default: `640.0`
- `center_x` (double): Camera principal point X coordinate
  - Default: `320.0` (image_width / 2)
- `center_y` (double): Camera principal point Y coordinate
  - Default: `240.0` (image_height / 2)

#### Gaze Computation
- `max_angle_threshold` (double): Maximum angle for "fully looking away" in degrees
  - Default: `90.0`
- `confidence_threshold` (double): Minimum confidence for facial landmarks
  - Default: `0.1`

#### Output
- `receiver_id` (string): Static receiver ID for gaze messages
  - Default: `"pin_hole_cam_model"`

#### Debug
- `enable_debug_output` (bool): Enable debug logging
  - Default: `true`
- `publish_rate` (double): Maximum publishing rate in Hz
  - Default: `30.0`


## Authors

- Josep Bravo (josep.bravo@eurecat.org)
