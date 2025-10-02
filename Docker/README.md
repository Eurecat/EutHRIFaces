# EutHRIFaces Docker Setup

This Docker setup provides a containerized environment for face detection and recognition capabilities within the HRI stack.

## Quick Start

1. **Build the container:**
   ```bash
   ./build_container.sh
   ```

2. **Run the container:**
   ```bash
   docker-compose up
   ```

## Build Options

- **Default (Vulcanexus):** `./build_container.sh`
- **Standard ROS2:** `./build_container.sh --standard-ros`
- **Clean rebuild:** `./build_container.sh --clean-rebuild [--standard-ros]`

## Features

- Face detection and recognition libraries (face-recognition, dlib, MediaPipe)
- ONNX Runtime support for optimized inference
- Deep learning frameworks (PyTorch, transformers)
- ROS2 Jazzy integration
- GPU support (when available)

## Dependencies

The container includes:
- **Face Processing:** face-recognition, dlib, MediaPipe, MTCNN, RetinaFace
- **Deep Learning:** PyTorch, transformers, InsightFace
- **Computer Vision:** OpenCV, scikit-image
- **ROS2:** Full Jazzy distribution with colcon build tools

## Volume Mounts

- Face detection packages will be mounted from `../face_detection` (when available)
- Face recognition packages will be mounted from `../face_recognition` (when available)
- X11 forwarding for GUI applications
- VSCode configuration for development

## Environment Variables

See `.env.example` for available configuration options. Copy it to `.env` and modify as needed.

## Development

The development compose file (`dev-docker-compose.yaml`) is identical to the main one but can be customized for development-specific needs.

## Notes

- The container runs with host networking for ROS2 communication
- GPU support is available when NVIDIA runtime is configured
- All face detection packages will be automatically built during container startup
