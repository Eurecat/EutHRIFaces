# Face Recognition Weights Directory

This directory is intended to store pre-trained face embedding model weights to avoid downloading them during runtime.

## Expected Files

Place the following files here based on your chosen model:

### VGGFace2 Model (default)
- `20180402-114759-vggface2.pt` - Pre-trained FaceNet model on VGGFace2 dataset

### CASIA-WebFace Model  
- `20180408-102900-casia-webface.pt` - Pre-trained FaceNet model on CASIA-WebFace dataset

## Configuration

The face recognition node will look for weights in this directory first. Configure the parameters:

```yaml
# In config/face_recognition.yaml
face_recognition_node:
  ros__parameters:
    weights_path: 'weights'  # Path to this directory
    face_embedding_weights_name: '20180402-114759-vggface2.pt'  # Specific weights file
    face_embedding_model: 'vggface2'  # Model type
```

## Downloading Weights

If weights are not found locally, the system will:
1. Download them from PyTorch model hub to cache
2. Copy them to this directory for future use (if `face_embedding_weights_name` is specified)

## Similar to YOLO Approach

This follows the same pattern as EUT YOLO:
- `weights_path` = directory containing weights files  
- `face_embedding_weights_name` = specific weights filename
- Similar to YOLO's `yolo_weights_path` + `yolo_weights_name`
