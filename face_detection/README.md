# Face Detection Package

This ROS2 package provides face detection capabilities using YOLO face detection model for HRI (Human-Robot Interaction) applications.

## Features

- **YOLO Face Detection**: Uses YOLOv8-based face detection model with 5 key facial landmarks
- **BOXMOT Tracking**: Optional face tracking using BOXMOT for consistent face IDs across frames
- **ROS4HRI Compatibility**: Publishes `hri_msgs/FacialLandmarks` messages following ros4hri standard  
- **Auto Model Download**: Automatically downloads the YOLO face model if not present
- **Configurable Parameters**: Adjustable confidence thresholds, device selection (CPU/GPU), etc.
- **Visualization Output**: Optional annotated image output for debugging and visualization

## Input/Output

### Input
- **Topic**: `/camera/color/image_rect_raw` (configurable)
- **Type**: `sensor_msgs/Image` (BGR format)

### Output
- **Topic**: `/face_detection/facial_landmarks` (configurable)  
- **Type**: `hri_msgs/FacialLandmarks`
- **Optional Visualization**: `/face_detection/image_with_faces` (annotated image)

## Landmarks Mapping

The YOLO face detector provides 5 key facial landmarks:
1. Left eye
2. Right eye  
3. Nose tip
4. Left mouth corner
5. Right mouth corner

These are mapped to the ros4hri `FacialLandmarks` message format (70 landmarks total) as follows:
- Left eye → `LEFT_EYE_INSIDE` (index 42)
- Right eye → `RIGHT_EYE_INSIDE` (index 39)
- Nose → `NOSE` (index 30)
- Left mouth → `MOUTH_OUTER_LEFT` (index 54)
- Right mouth → `MOUTH_OUTER_RIGHT` (index 48)

All other landmarks are set as invalid (confidence = 0.0).

## Installation

1. Ensure you have ROS2 and the required dependencies installed
2. Clone this repository into your workspace
3. Build the package:
   ```bash
   cd your_ws
   colcon build --packages-select face_detection
   source install/setup.bash
   ```

## Usage

### Launch the node
```bash
ros2 launch face_detection face_detection.launch.py
```

### With custom parameters
```bash
ros2 launch face_detection face_detection.launch.py \
    input_topic:=/your/camera/topic \
    device:=cuda \
    confidence_threshold:=0.7
```

### Run node directly
```bash
ros2 run face_detection face_detector
```

## Configuration

The package can be configured via:
1. Launch file parameters
2. Configuration YAML file (`config/face_detection.yaml`)
3. ROS2 parameters at runtime

### Key Parameters

- `input_topic`: Input camera topic (default: `/camera/color/image_rect_raw`)
- `output_topic`: Output landmarks topic (default: `/face_detection/facial_landmarks`)
- `model_path`: Path to YOLO model file (default: `weights/yolov8n-face.onnx`)
- `confidence_threshold`: Minimum face detection confidence (default: 0.5)
- `iou_threshold`: IoU threshold for NMS (default: 0.4)
- `device`: Inference device - `cpu` or `cuda` (default: `cpu`)
- `enable_image_output`: Enable visualization output (default: `true`)
- `use_boxmot`: Enable BOXMOT tracking for consistent face IDs (default: `false`)
- `boxmot_tracker_type`: Type of BOXMOT tracker - `bytetrack`, `botsort`, `strongsort`, etc. (default: `bytetrack`)
- `boxmot_reid_model`: Path to ReID model for improved tracking (optional)

## Model

The package uses YOLOv8n-face model which provides:
- Fast inference suitable for real-time applications
- Accurate face detection with 5 key facial landmarks
- ONNX format for cross-platform compatibility

The model is automatically downloaded from:
https://raw.githubusercontent.com/hpc203/yolov8-face-landmarks-opencv-dnn/main/weights/yolov8n-face.onnx

## BOXMOT Tracking

The package supports optional face tracking using BOXMOT for consistent face IDs across video frames.

### Benefits of Tracking
- **Consistent Face IDs**: Same person gets the same face ID across frames
- **Temporal Consistency**: Reduces ID flickering in video sequences  
- **Improved Performance**: Better handling of occlusions and temporary face disappearances

### Enabling Tracking
Set `use_boxmot: true` in the configuration file or launch parameters:

```bash
ros2 launch face_detection face_detection.launch.py use_boxmot:=true
```

### Available Trackers
- `bytetrack` (default): Fast and efficient tracker
- `botsort`: ByteTrack with ReID features
- `strongsort`: Strong tracker with deep features
- `deepocsort`: DeepOCSORT tracker

### ReID Models (Optional)
For improved tracking across occlusions, you can specify a ReID model:
```yaml
boxmot_reid_model: "path/to/reid_model.pt"
```

## Dependencies

- ROS2 (tested on Humble/Iron)
- OpenCV (`python3-opencv`)
- NumPy (`python3-numpy`)
- ONNXRuntime (`python3-onnxruntime` or `onnxruntime-gpu`)
- hri_msgs (ros4hri message definitions)
- cv_bridge
- sensor_msgs
- std_msgs
- **BOXMOT** (`boxmot>=13.0.0`) - Optional for tracking features

## GPU Support

For GPU acceleration:
1. Install CUDA and cuDNN
2. Install onnxruntime-gpu: `pip install onnxruntime-gpu`
3. Set device parameter to `cuda`

## Troubleshooting

### Model Download Issues
- Ensure internet connectivity
- Check firewall settings
- Manually download model to `weights/` directory if needed

### GPU Issues
- Verify CUDA installation: `nvidia-smi`
- Check onnxruntime-gpu installation: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
- Fall back to CPU if GPU unavailable

### Performance Tips
- Use GPU for better performance when available
- Adjust `confidence_threshold` based on your use case
- Lower resolution input images for faster processing

## License

Apache-2.0

## Maintainer

Josep Bravo (josep.bravo@eurecat.org)
