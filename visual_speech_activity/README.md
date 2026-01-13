# Enhanced Visual Speech Activity Detection

A ROS2 package for visual-only speech activity detection (V-VAD) using advanced lip movement analysis. This package extends the HRI faces pipeline with real-time speaking detection based on comprehensive temporal analysis of facial landmarks.

## Overview

This package provides robust visual speech activity detection by:
- Analyzing temporal patterns in full 68-point dlib facial landmarks (with 5-point fallback)
- Leveraging `recognized_face_id` from face recognition for stable tracking
- Using RNN-based temporal classification with multiple lip movement features
- Computing speaking confidence scores using advanced geometric features
- Supporting both ROS4HRI array mode and per-ID topic modes

**Key Features:**
- ✅ **Enhanced accuracy** - Full 68-point dlib landmark support with RNN classification
- ✅ **Visual-only detection** - No audio required
- ✅ **Real-time performance** - Optimized for live video streams
- ✅ **Robust** - Uses `recognized_face_id` for stable tracking across frame changes
- ✅ **Configurable** - Adjustable sensitivity and temporal window parameters
- ✅ **ROS4HRI compliant** - Extends standard `FacialRecognition` message

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────────┐
│ Face Detection  │─────>│ Face Recognition │─────>│ Visual Speech Activity  │
│ (68 landmarks)  │      │  (recognized_id) │      │   (speaking detection)  │
└─────────────────┘      └──────────────────┘      └─────────────────────────┘
  dlib 68-points or       Stable identity IDs        is_speaking + confidence
  YOLO 5-points (fallback)                           RNN-based classification
```

### Pipeline Integration

1. **face_detection** publishes `FacialLandmarks` with full 68-point dlib landmarks (or 5-point fallback)
2. **face_recognition** publishes `FacialRecognition` with stable `recognized_face_id`
3. **visual_speech_activity** (this package):
   - Subscribes to both landmarks and recognition messages
   - Maintains temporal buffers per `recognized_face_id` (default: 20 frames)
   - Extracts comprehensive lip features (MAR, MER, lip height/width)
   - Uses RNN-based temporal classification for robust detection
   - Publishes extended `FacialRecognition` with `is_speaking` and `speaking_confidence`

## Enhanced Algorithm

The detection uses an enhanced **Lip-Movement-Net** inspired approach with multiple improvements:

### 1. Feature Extraction
- **Full dlib landmarks**: Uses points 48-67 (mouth region) for precise lip analysis
- **Multiple features**: MAR, MER, inner lip height/width, outer lip contours
- **Fallback support**: Automatic fallback to 5-point landmarks when dlib unavailable

### 2. Temporal Classification
- **RNN-based**: Simple LSTM-like network for temporal pattern recognition
- **Per-identity buffers**: Maintains separate hidden states per `recognized_face_id`
- **Adaptive**: Learns from temporal patterns rather than simple thresholding

### 3. Feature Details
- **MAR (Mouth Aspect Ratio)**: Vertical mouth opening / horizontal width
- **MER (Mouth Elongation Ratio)**: Horizontal mouth stretching
- **Lip Height**: Inner lip vertical distance (normalized)
- **Lip Width**: Inner lip horizontal distance (normalized)
5. **Classification**: Weighted combination of features → confidence score
6. **Temporal Smoothing**: Voting over last 5 decisions to reduce jitter

**Why this approach?**
- Simple, interpretable, and computationally efficient
- No heavy neural networks (suitable for real-time embedded systems)
- Uses existing 5-point landmarks (no additional detectors needed)
- Proven effective for visual-only speech activity detection

## Installation

### Dependencies

This package requires:
- ROS2 (Humble or later)
- Python 3.8+
- `hri_msgs` (for ROS4HRI message definitions)
- NumPy
- OpenCV (cv_bridge)
- SciPy (optional, for advanced filtering)

### Building

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# If not already cloned, clone the EutHRIFaces repository
git clone <repository_url>

# Build the package
cd ~/ros2_ws
colcon build --packages-select visual_speech_activity

# Source the workspace
source install/setup.bash
```

## Usage

### Quick Start (Array Mode)

```bash
# Launch the visual speech activity node
ros2 launch visual_speech_activity visual_speech_activity.launch.py

# The node will subscribe to:
#   /humans/faces/detected (FacialLandmarksArray)
#   /humans/faces/recognized (FacialRecognitionArray)
# 
# And publish to:
#   /humans/faces/speaking (FacialRecognitionArray with speaking fields)
```

### Per-ID Mode (ROS4HRI Standard)

```bash
# Launch with per-ID topic mode
ros2 launch visual_speech_activity visual_speech_activity.launch.py ros4hri_with_id:=true

# The node will subscribe to:
#   /humans/faces/tracked (IdsList)
#   /humans/faces/<face_id>/landmarks (FacialLandmarks per face)
#   /humans/faces/<recognized_id>/recognition (FacialRecognition per identity)
#
# And publish to:
#   /humans/faces/<recognized_id>/speaking (FacialRecognition with speaking)
```

### Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recognition_input_topic` | string | `/humans/faces/recognized` | Input topic for recognition messages |
| `landmarks_input_topic` | string | `/humans/faces/detected` | Input topic for landmarks messages |
| `output_topic` | string | `/humans/faces/speaking` | Output topic for speaking detection |
| `ros4hri_with_id` | bool | `false` | Use per-ID topics instead of arrays |
| `window_size` | int | `20` | Temporal buffer size (frames) |
| `movement_threshold` | float | `0.02` | Minimum MAR variation threshold |
| `speaking_threshold` | float | `0.5` | Confidence threshold for classification |
| `temporal_smoothing` | bool | `true` | Enable temporal smoothing |
| `min_frames_for_detection` | int | `5` | Min frames before detection starts |
| `enable_debug_output` | bool | `false` | Enable debug logging |

### Example: Custom Configuration

```bash
ros2 launch visual_speech_activity visual_speech_activity.launch.py \
    window_size:=30 \
    speaking_threshold:=0.6 \
    enable_debug_output:=true
```

## Output Message Format

The node publishes extended `FacialRecognition` messages with these additional fields:

```
std_msgs/Header header
string face_id                  # Volatile tracking ID
string recognized_face_id       # Stable recognized identity ID
float32 confidence              # Recognition confidence

# Visual Speech Activity Detection (new fields)
bool is_speaking                # True if person is speaking
float32 speaking_confidence     # Confidence of speaking [0.0-1.0]
```

### Example Message

```yaml
header:
  stamp: {sec: 1234567890, nanosec: 123456789}
face_id: "face_0"
recognized_face_id: "person_abc123"
confidence: 0.95
is_speaking: true
speaking_confidence: 0.78
```

## Configuration Tuning

### For High Precision (Fewer False Positives)

```yaml
speaking_threshold: 0.6        # Higher threshold
movement_threshold: 0.03       # Higher sensitivity requirement
window_size: 30                # Longer temporal analysis
temporal_smoothing: true       # Enable smoothing
```

### For High Recall (Catch More Speaking)

```yaml
speaking_threshold: 0.4        # Lower threshold
movement_threshold: 0.015      # Lower sensitivity requirement
window_size: 15                # Shorter response time
min_frames_for_detection: 3    # Start detection earlier
```

### For Real-Time / Low Latency

```yaml
window_size: 10                # Smaller buffer
min_frames_for_detection: 3    # Quick startup
temporal_smoothing: false      # Disable smoothing (faster but noisier)
```

## Testing

### Visualize Speaking Detection

```bash
# Terminal 1: Run the full pipeline
ros2 launch visual_speech_activity visual_speech_activity.launch.py enable_debug_output:=true

# Terminal 2: Echo speaking results
ros2 topic echo /humans/faces/speaking

# You should see messages with is_speaking and speaking_confidence fields
```

### Monitor Performance

```bash
# Check processing rate
ros2 topic hz /humans/faces/speaking

# Check latency
ros2 topic delay /humans/faces/speaking
```

## Integration with Other Nodes

### Complete HRI Pipeline

```bash
# 1. Start camera
ros2 launch realsense2_camera rs_launch.py

# 2. Start face detection
ros2 launch face_detection face_detection.launch.py

# 3. Start face recognition
ros2 launch face_recognition face_recognition.launch.py

# 4. Start visual speech activity detection
ros2 launch visual_speech_activity visual_speech_activity.launch.py
```

### Using in Your Node (Python)

```python
from rclpy.node import Node
from hri_msgs.msg import FacialRecognitionArray

class MySpeechAwareNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.sub = self.create_subscription(
            FacialRecognitionArray,
            '/humans/faces/speaking',
            self.speaking_callback,
            10
        )
    
    def speaking_callback(self, msg):
        for recognition in msg.recognitions:
            if recognition.is_speaking:
                self.get_logger().info(
                    f"Person {recognition.recognized_face_id} is speaking "
                    f"(confidence: {recognition.speaking_confidence:.2f})"
                )
```

## Troubleshooting

### No Speaking Detected

1. **Check landmarks quality**: Ensure face_detection is publishing good landmarks
   ```bash
   ros2 topic echo /humans/faces/detected
   ```

2. **Lower threshold**: Try reducing `speaking_threshold` to 0.4 or 0.3

3. **Check buffer size**: Ensure enough frames are being processed
   ```bash
   ros2 param get /visual_speech_activity_node window_size
   ```

### Too Many False Positives

1. **Increase threshold**: Raise `speaking_threshold` to 0.6 or 0.7
2. **Enable smoothing**: Set `temporal_smoothing:=true`
3. **Increase movement threshold**: Raise `movement_threshold` to 0.03-0.05

### High Latency

1. **Reduce window size**: Try `window_size:=10`
2. **Disable smoothing**: Set `temporal_smoothing:=false`
3. **Check CPU usage**: Visual detection should be lightweight (<5% CPU)

## Limitations

- **Visual-only**: Cannot detect speaking if mouth is occluded or face is not frontal
- **Landmark quality**: Depends on quality of face detection landmarks
- **No audio**: Cannot detect non-lip-moving speech (e.g., ventriloquism)
- **Approximation**: Uses only 2 mouth keypoints (corners), not full lip contour

## Future Enhancements

- [ ] Add dlib 68-point landmarks support for more accurate lip detection
- [ ] Implement mouth region cropping for CNN-based detection
- [ ] Add audio-visual fusion when microphone available
- [ ] Support for profile/side-view faces
- [ ] Multi-modal confidence fusion with audio VAD

## References

- **Lip-Movement-Net**: https://github.com/sachinsdate/lip-movement-net
- **ROS4HRI Standard**: https://wiki.ros.org/hri
- **Mouth Aspect Ratio (MAR)**: Based on Eye Aspect Ratio (EAR) for drowsiness detection

## License

Apache-2.0

## Maintainer

Josep Bravo (josep.bravo@eurecat.org)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{visual_speech_activity_ros2,
  author = {Bravo, Josep},
  title = {Visual Speech Activity Detection for ROS2},
  year = {2026},
  publisher = {Eurecat},
  url = {https://github.com/Eurecat/EutHRIFaces}
}
```
