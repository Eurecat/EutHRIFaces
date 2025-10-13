# Face Recognition Package

This ROS2 package provides face recognition capabilities for HRI (Human-Robot Interaction) applications using face embeddings and identity management. The approach is 100% based on the EUT YOLO identity management system, providing persistent identity tracking across changing track IDs.

## Features

- **Face Embedding Extraction**: Uses FaceNet models (VGGFace2 or CASIA-WebFace) for high-quality face embeddings
- **Identity Management**: Persistent identity tracking across frames using clustering and similarity matching
- **Temporal Tracking**: Maintains identities even when track IDs change
- **Batch Processing**: Optimized batch processing for better performance
- **Persistent Storage**: Optional identity database for maintaining identities across sessions
- **ROS4HRI Compatibility**: Publishes `FacialRecognition` messages following the ros4hri standard

## Architecture

The package follows the EUT YOLO identity management approach with these key components:

### Face Embedding Extractor
- Extracts 512-dimensional face embeddings using pre-trained FaceNet models
- Supports both CPU and GPU inference
- Batch processing for optimal performance

### Identity Manager
- Manages unique persistent IDs (U1, U2, etc.) for detected faces
- Uses cosine similarity and clustering for identity assignment
- Handles identity merging and lifecycle management
- Maintains embedding statistics and quality scores

### Face Recognition Node
- Main ROS2 node that orchestrates the recognition pipeline
- Subscribes to `FacialLandmarksArray` messages from face detection
- Publishes `FacialRecognition` messages with persistent identity assignments

## Input/Output

### Input Topics
- `/humans/faces/detected` (hri_msgs/FacialLandmarksArray): Face detection results with bounding boxes and landmarks
- `/camera/color/image_rect_raw` (sensor_msgs/Image): RGB camera images for face cropping

### Output Topics
- `/humans/faces/recognized` (hri_msgs/FacialRecognition): Recognition results with persistent unique IDs

## Installation

1. Ensure you have ROS2 Humble and the required dependencies:
```bash
# Install ROS2 dependencies
sudo apt install ros-humble-hri-msgs ros-humble-cv-bridge

# Install Python dependencies
pip install torch torchvision facenet-pytorch opencv-python numpy scikit-learn scipy Pillow PyYAML
```

2. Clone this package into your ROS2 workspace:
```bash
cd ~/your_ws/src
# Package should already be in EutHRIFaces/face_recognition
```

3. Build the workspace:
```bash
cd ~/your_ws
colcon build --packages-select face_recognition
source install/setup.bash
```

## Usage

### Basic Usage

Launch the face recognition node:
```bash
ros2 launch face_recognition face_recognition.launch.py
```

### With Custom Parameters

```bash
ros2 launch face_recognition face_recognition.launch.py \
    input_topic:=/your/face/detection/topic \
    device:=cuda \
    similarity_threshold:=0.7 \
    enable_debug_prints:=true
```

### Run Node Directly

```bash
ros2 run face_recognition face_recognition_node
```

## Configuration

Key parameters can be configured in `config/face_recognition.yaml` or via launch arguments:

### Face Embedding Parameters
- `face_embedding_model`: Model to use ('vggface2' or 'casia-webface')
- `device`: Device for inference ('cpu' or 'cuda')
- `weights_path`: Path to pre-downloaded model weights (optional)

### Identity Management Parameters (from EUT YOLO)
- `similarity_threshold`: Minimum similarity for identity assignment (default: 0.6)
- `clustering_threshold`: Threshold for clustering embeddings (default: 0.7)
- `max_embeddings_per_identity`: Maximum embeddings stored per identity (default: 50)
- `identity_timeout`: Seconds before inactive identity removal (default: 60.0)
- `track_identity_stickiness_margin`: Margin for track identity persistence (default: 0.4)
- `embedding_inclusion_threshold`: Threshold for including embeddings in cluster (default: 0.6)

## Identity Management System

The identity management system is directly adapted from EUT YOLO and provides:

### Unique Identity Assignment
- Assigns persistent unique IDs (U1, U2, U3, etc.) to detected faces
- Maintains identities across changing track IDs from face detection
- Uses face embedding similarity for re-identification

### Clustering and Similarity
- Uses cosine similarity between face embeddings
- Clusters embeddings belonging to the same person
- Handles identity merging when multiple track IDs belong to same person

### Quality Scoring
- Calculates quality scores based on embedding consistency
- Tracks detection counts and temporal stability
- Filters out poor quality faces based on gaze scores

### Persistent Storage
- Optional JSON database for maintaining identities across sessions
- Automatic loading/saving of identity database
- Configurable database path

## Message Format

The package publishes `hri_msgs/FacialRecognition` messages:

```
std_msgs/Header header
string face_id              # Original face ID from detection
string recognized_face_id   # Persistent unique ID (U1, U2, etc.)
float32 confidence         # Cosine similarity confidence score
```

## Performance Optimization

The package includes several optimizations from the EUT YOLO implementation:

- **Batch Processing**: Groups multiple faces for efficient GPU utilization
- **Embedding Caching**: Reuses embeddings when possible
- **Quality Filtering**: Excludes poor quality faces from processing
- **EWMA Updates**: Optional exponentially weighted moving averages for mean embeddings
- **Configurable Timeouts**: Balances latency vs. batch efficiency

## Integration with EutHRIFaces Pipeline

This package is designed to work seamlessly with the EutHRIFaces pipeline:

1. **Face Detection** → Detects faces and publishes `FacialLandmarksArray`
2. **Gaze Estimation** → Estimates gaze and publishes `Gaze` messages  
3. **Face Recognition** → Assigns persistent identities and publishes `FacialRecognition`

## Dependencies

- ROS2 Humble
- hri_msgs
- OpenCV (cv2)
- NumPy
- PyTorch
- torchvision
- facenet-pytorch
- scikit-learn
- scipy
- Pillow
- PyYAML

## Troubleshooting

### Common Issues

1. **"facenet-pytorch not available"**
   ```bash
   pip install facenet-pytorch
   ```

2. **"CUDA out of memory"**
   - Set `device: 'cpu'` in configuration

3. **"No image available for face recognition"**
   - Ensure image topic is publishing
   - Check topic names match between face detection and recognition

4. **Poor recognition performance**
   - Increase `similarity_threshold`
   - Adjust `clustering_threshold`
   - Enable debug prints to analyze similarities

### Debug Mode

Enable detailed debug output:
```bash
ros2 launch face_recognition face_recognition.launch.py enable_debug_prints:=true
```

This will show:
- Embedding similarities for each face
- Identity assignment decisions
- Clustering operations
- Performance metrics

## License

Apache-2.0

## Maintainer

Josep Bravo (josep.bravo@eurecat.org)
