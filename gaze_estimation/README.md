# Gaze Estimation Package

This ROS2 package provides gaze estimation capabilities for Human-Robot Interaction (HRI) applications. It subscribes to facial landmarks from face detection and computes gaze direction and confidence scores using a pinhole camera model.

## Overview

The gaze estimation node:
- Subscribes to `FacialLandmarks` messages from face detection
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
    input_topic:=/my_face_detection/facial_landmarks \
    output_topic:=/my_gaze_estimation/gaze
```

## Configuration

### Parameters

The node accepts the following parameters:

#### Topics
- `input_topic` (string): Input topic for FacialLandmarks messages
  - Default: `/face_detection/facial_landmarks`
- `output_topic` (string): Output topic for Gaze messages  
  - Default: `/gaze_estimation/gaze`

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

### Configuration File

Edit `config/gaze_estimation.yaml` to modify default parameters:

```yaml
gaze_estimation_node:
  ros__parameters:
    # Camera parameters
    focal_length: 800.0
    image_width: 1280
    image_height: 720
    center_x: 640.0
    center_y: 360.0
    
    # Gaze computation
    max_angle_threshold: 80.0
    confidence_threshold: 0.2
    
    # Debug
    enable_debug_output: false
```

## Algorithm

The gaze estimation algorithm follows these steps:

1. **Landmark Extraction**: Extract 5 key landmarks from ros4hri FacialLandmarks:
   - Nose tip (NOSE = 30)
   - Right eye (RIGHT_PUPIL = 68 or RIGHT_EYE_INSIDE = 39)
   - Left eye (LEFT_PUPIL = 69 or LEFT_EYE_INSIDE = 42)  
   - Right lip corner (MOUTH_OUTER_RIGHT = 48)
   - Left lip corner (MOUTH_OUTER_LEFT = 54)

2. **3D Pose Estimation**: Use OpenCV's solvePnP with a generic 3D face model:
   - Nose tip at origin (0, 0, 0)
   - Eyes and mouth positioned relative to nose
   - Solve for head rotation and translation

3. **Gaze Direction**: Extract gaze direction from head pose rotation matrix:
   - Gaze direction = -Z axis of head coordinate system
   - Represents the forward-looking direction

4. **Gaze Score**: Compute confidence based on head orientation:
   - Score = 1.0 when looking straight at camera (yaw=0°, pitch=0°)
   - Score decreases as head turns away
   - Score = 0.0 when head is turned beyond max_angle_threshold

## Message Types

### Input: hri_msgs/FacialLandmarks
```
std_msgs/Header header
string face_id
NormalizedPointOfInterest2D[] landmarks
uint32 height
uint32 width
float32 bbox_confidence
uint32[] bbox_xyxy
float32[] bbox_centroid
```

### Output: hri_msgs/Gaze
```
std_msgs/Header header
string sender      # face_id from input
string receiver    # "pin_hole_cam_model"
float32 score      # Gaze confidence [0.0, 1.0]
geometry_msgs/Vector3 gaze_direction  # 3D gaze vector
```

## Troubleshooting

### No Gaze Messages Published
- Check that input topic has FacialLandmarks messages: `ros2 topic echo /face_detection/facial_landmarks`
- Verify required landmarks are present in the messages
- Check debug output for error messages

### Low Gaze Scores
- Adjust `max_angle_threshold` parameter to be more/less restrictive
- Verify camera parameters match your actual camera setup
- Check facial landmark quality from face detection

### Performance Issues
- Adjust `publish_rate` parameter to limit processing frequency
- Disable `enable_debug_output` for production use
- Check CPU usage with `top` or `htop`

## Development

### Running Tests
```bash
cd ~/ros2_ws
colcon test --packages-select gaze_estimation
```

### Code Structure
- `gaze_estimation_node.py`: Main ROS2 node
- `gaze_utils.py`: Gaze computation utilities
- `config/gaze_estimation.yaml`: Default configuration
- `launch/gaze_estimation.launch.py`: Launch file

### Adding Custom Face Models
Modify the `model_points` parameter in the config file to use different 3D face models:

```yaml
model_points:
  nose_tip: [0.0, 0.0, 0.0]
  right_eye: [25.0, -35.0, -25.0]  # Custom positions
  left_eye: [-25.0, -35.0, -25.0]
  # ... etc
```

## License

Apache-2.0

## Authors

- Josep Bravo (josep.bravo@eurecat.org)
