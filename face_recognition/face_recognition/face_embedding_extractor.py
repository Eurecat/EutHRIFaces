"""
Face embedding extraction utility for face recognition package.

This utility provides methods to extract face embeddings from detected face crops
using pre-trained models like FaceNet. It's directly adapted from the EUT YOLO
implementation to maintain consistency and compatibility.
"""

from datetime import datetime
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List
import os
from pathlib import Path

try:
    import torch
    from PIL import Image
    # from torchvision import transforms
    from torchvision.transforms import v2 as transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. Face embedding extraction will be disabled.")

try:
    from facenet_pytorch import InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False
    print("[WARNING] facenet-pytorch not available. Install with: pip install facenet-pytorch")


class FaceEmbeddingExtractor:
    """
    Utility class for extracting face embeddings from face crops.
    
    This class handles the preprocessing and inference for face embedding extraction,
    providing a simple interface to get embeddings from detected face regions.
    Directly adapted from EUT YOLO implementation.
    """
    
    def __init__(self, model_name: str = "vggface2", device: Optional[str] = None, 
                 input_size: Tuple[int, int] = (160, 160), weights_path: Optional[str] = None,
                 face_embedding_weights_path: Optional[str] = None):
        """
        Initialize the face embedding extractor.
        
        Args:
            model_name: Pre-trained model to use ("vggface2" or "casia-webface")
            device: Device to run inference on (auto-detect if None)
            input_size: Input size for the model (width, height)
            weights_path: Path to directory containing pre-downloaded weights (to avoid downloading)
            face_embedding_weights_path: Specific path to face embedding weights file (for copying after download)
        """
        self.model_name = model_name
        self.input_size = input_size
        self.weights_path = weights_path
        self.face_embedding_weights_path = face_embedding_weights_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() and _TORCH_AVAILABLE else "cpu"
        else:
            self.device = device
        light_green = "\033[38;5;82m"
        reset = "\033[0m"
        print(f"{light_green}[INFO] Using device for face embedding: {self.device}{reset}")
        # Model components
        self.model = None
        self.transform = None
        
        # Initialize if dependencies are available
        if _TORCH_AVAILABLE and _FACENET_AVAILABLE:
            self._setup_model()
            self._setup_transforms()
        else:
            print("[WARNING] Face embedding dependencies not available")
    
    def _setup_model(self) -> None:
        """Setup the face embedding model."""
        try:
            print(f"[INFO] Setting up face embedding model: {self.model_name}")
            
            # Ensure weights are available (copy from local if available)
            self._ensure_weights_available()
            
            # Initialize the model
            self.model = InceptionResnetV1(pretrained=self.model_name).eval().to(self.device)
            self.model.eval()
            
            print(f"\033[92m[INFO] Face embedding model loaded successfully on {self.device}\033[0m")
            
            # Copy weights to configured path if specified
            if self.face_embedding_weights_path:
                self._copy_weights_from_cache_to_configured_path()
                
        except Exception as e:
            print(f"[ERROR] Failed to setup face embedding model: {e}")
            self.model = None
    
    def _ensure_weights_available(self) -> bool:
        """
        Ensure weights are available in torch cache directory to avoid downloading.
        Copies weights from configured weights_path to torch cache if needed.
        
        Returns:
            bool: True if weights were copied to cache, False otherwise
        """
        try:
            from torch.hub import get_dir as get_torch_home
            import shutil
            
            # If no weights_path is provided, just download normally
            if not self.weights_path:
                print(f"[INFO] No local weights path provided, will download {self.model_name} model if needed")
                # temp_device = "cuda"  # Use CPU for initial weight loading
                temp_model = InceptionResnetV1(pretrained=self.model_name, device=self.device)
                del temp_model  # Clean up temporary model
                return True
            
            # Get torch cache directory
            torch_home = get_torch_home()
            # Remove '/hub' from torch_home if present, then add 'checkpoints'
            checkpoints_dir = torch_home.replace('/hub', '')
            checkpoints_dir = os.path.join(checkpoints_dir, 'checkpoints')
            
            # Define weight file mapping
            weight_files = {
                'vggface2': '20180402-114759-vggface2.pt',
                'casia-webface': '20180408-102900-casia-webface.pt'
            }
            
            weight_filename = weight_files.get(self.model_name)
            if not weight_filename:
                print(f"[WARNING] Unknown model name {self.model_name}, weights may be downloaded")
                return False
            
            # Paths
            source_path = os.path.join(self.weights_path, weight_filename)
            target_path = os.path.join(checkpoints_dir, weight_filename)
            
            # Create checkpoints directory if it doesn't exist
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # Copy weights if source exists and target doesn't exist or is smaller
            if os.path.exists(source_path):
                source_size = os.path.getsize(source_path)
                
                if not os.path.exists(target_path):
                    print(f"[INFO] Copying face embedding weights from {source_path} to {target_path}")
                    shutil.copy2(source_path, target_path)
                    print(f"[INFO] Weights copied successfully ({source_size} bytes)")
                    return True
                else:
                    target_size = os.path.getsize(target_path)
                    if target_size != source_size:
                        print(f"[INFO] Updating face embedding weights (size mismatch: {target_size} vs {source_size})")
                        shutil.copy2(source_path, target_path)
                        print(f"[INFO] Weights updated successfully")
                        return True
                    else:
                        print(f"[INFO] Face embedding weights already available in cache ({target_size} bytes)")
                        return False
            else:
                print(f"[INFO] Pre-downloaded weights not found at {source_path}")
                print(f"[INFO] Model will be downloaded from internet and then copied to configured path")
                # Download the model first
                try:
                    temp_device = "cpu"  # Use CPU for initial weight loading
                    temp_model = InceptionResnetV1(pretrained=self.model_name, device=temp_device)
                    del temp_model  # Clean up temporary model
                    print(f"[INFO] Model {self.model_name} downloaded successfully")
                    return True
                except Exception as download_e:
                    print(f"[ERROR] Failed to download model {self.model_name}: {download_e}")
                    return False
                
        except Exception as e:
            print(f"[WARNING] Failed to copy weights to cache: {e}")
            print(f"[WARNING] Model may be downloaded from internet")
            return False
    
    def _copy_weights_from_cache_to_configured_path(self) -> None:
        """
        Copy downloaded weights from torch cache to configured path for future use.
        """
        try:
            from torch.hub import get_dir as get_torch_home
            import shutil
            
            if not self.face_embedding_weights_path:
                return
            
            # Get torch cache directory
            torch_home = get_torch_home()
            # Remove '/hub' from torch_home if present, then add 'checkpoints'
            checkpoints_dir = torch_home.replace('/hub', '')
            checkpoints_dir = os.path.join(checkpoints_dir, 'checkpoints')
            
            # Define weight file mapping
            weight_files = {
                'vggface2': '20180402-114759-vggface2.pt',
                'casia-webface': '20180408-102900-casia-webface.pt'
            }
            
            weight_filename = weight_files.get(self.model_name)
            if not weight_filename:
                print(f"[WARNING] Unknown model name {self.model_name}, cannot copy weights")
                return
            
            # Paths
            source_path = os.path.join(checkpoints_dir, weight_filename)
            target_path = self.face_embedding_weights_path
            
            # Copy weights if source exists and target doesn't exist or is smaller
            if os.path.exists(source_path):
                source_size = os.path.getsize(source_path)
                
                # Create target directory if it doesn't exist
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                if not os.path.exists(target_path):
                    print(f"[INFO] Copying downloaded face embedding weights from cache to configured path:")
                    print(f"[INFO] From: {source_path}")
                    print(f"[INFO] To: {target_path}")
                    shutil.copy2(source_path, target_path)
                    print(f"[INFO] Weights copied to configured path ({source_size} bytes)")
                else:
                    target_size = os.path.getsize(target_path)
                    if target_size != source_size:
                        print(f"[INFO] Updating weights in configured path (size mismatch: {target_size} vs {source_size})")
                        shutil.copy2(source_path, target_path)
                        print(f"[INFO] Weights updated in configured path")
                    else:
                        print(f"[INFO] Weights already up-to-date in configured path ({target_size} bytes)")
            else:
                print(f"[WARNING] Source weights not found in cache: {source_path}")
                
        except Exception as e:
            print(f"[WARNING] Failed to copy weights to configured path: {e}")
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        try:
            self.transform = transforms.Compose([
                transforms.ToImage(),  # Converts PIL or numpy to tensor
                transforms.Resize(self.input_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
            print("[INFO] Face embedding transforms initialized")
        except Exception as e:
            print(f"[ERROR] Failed to setup transforms: {e}")
            self.transform = None
    
    def is_available(self) -> bool:
        """Check if the embedding extractor is available for use."""
        return (self.model is not None and 
                self.transform is not None and 
                _TORCH_AVAILABLE and 
                _FACENET_AVAILABLE)
    
    def crop_face_from_image(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                           scaling_factor: float = 1.0) -> Optional[np.ndarray]:
        """
        Crop face region from image using bounding box.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_bbox: Face bounding box as (x, y, width, height)
            scaling_factor: Factor to scale the bounding box (default: 1.0)
            
        Returns:
            Cropped face image or None if cropping fails
        """
        try:
            x, y, w, h = face_bbox
            
            # Apply scaling factor
            if scaling_factor != 1.0:
                center_x, center_y = x + w // 2, y + h // 2
                w_scaled = int(w * scaling_factor)
                h_scaled = int(h * scaling_factor)
                x = center_x - w_scaled // 2
                y = center_y - h_scaled // 2
                w, h = w_scaled, h_scaled
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
                
            # Crop the face region
            face_crop = image[y:y+h, x:x+w]
            return face_crop
            
        except Exception as e:
            print(f"Error cropping face from image: {e}")
            return None
            
    def preprocess_face_image(self, face_image: np.ndarray) -> Optional[torch.Tensor]:
        if face_image is None or face_image.size == 0:
            return None
        try:
            # Convert BGR to RGB (still CPU, but minimal cost)
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0  # Convert to CHW float
            tensor = tensor.unsqueeze(0).to(self.device)

            # Apply transforms on GPU
            tensor = self.transform(tensor)
            return tensor
        except Exception as e:
            print(f"Error preprocessing face image: {e}")
            return None

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a face image.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        if not self.is_available():
            return None
            
        # Preprocess the image
        preprocessed = self.preprocess_face_image(face_image)
        if preprocessed is None:
            return None
            
        try:
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(preprocessed)
            
            # Convert to numpy and flatten
            embedding_np = embedding.cpu().detach().numpy().flatten()
            return embedding_np
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def extract_embedding_from_bbox(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                  scaling_factor: float = 1.0) -> Optional[np.ndarray]:
        """
        Extract face embedding directly from image and bounding box.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_bbox: Face bounding box as (x, y, width, height)
            scaling_factor: Factor to scale the bounding box (default: 1.0)
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        # Crop face from image
        face_crop = self.crop_face_from_image(image, face_bbox, scaling_factor)
        if face_crop is None:
            return None
            
        # Extract embedding from cropped face
        return self.extract_embedding(face_crop)
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray], gaze_scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """
        Extract face embeddings from multiple face images in a batch for better performance.
        
        Args:
            face_images: List of face images as numpy arrays (BGR format)
            gaze_scores: List of gaze scores corresponding to each face image (optional)

        Returns:
            List of face embeddings as numpy arrays or None for failed extractions
        """
        if not self.is_available():
            return [None] * len(face_images)
            
        if not face_images:
            return []
        
        try:
            # Preprocess all images
            preprocessed_batch = []
            valid_indices = []

            for i, face_image in enumerate(face_images):
                # Preprocess face image
                preprocessed = self.preprocess_face_image(face_image)
                if preprocessed is not None:
                    preprocessed_batch.append(preprocessed)
                    valid_indices.append(i)
            
            if not preprocessed_batch:
                return [None] * len(face_images)
            
            # Stack into batch tensor and ensure it's on the correct device
            batch_tensor = torch.cat(preprocessed_batch, dim=0).to(self.device)
            
            # Verify device placement for debugging
            if "cuda" in self.device and not batch_tensor.is_cuda:
                print(f"\033[93m[WARNING] Batch tensor not on GPU! Device: {self.device}, Tensor device: {batch_tensor.device}\033[0m")
            
            # Extract embeddings for the entire batch
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
            
            # Convert to numpy and create result list
            batch_embeddings_np = batch_embeddings.cpu().detach().numpy()
            results = [None] * len(face_images)
            
            for batch_idx, original_idx in enumerate(valid_indices):
                results[original_idx] = batch_embeddings_np[batch_idx].flatten()
            
            return results
            
        except Exception as e:
            print(f"Error extracting face embeddings in batch: {e}")
            return [None] * len(face_images)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity value between -1 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Ensure both embeddings are flattened
        emb1 = embedding1.flatten()
        emb2 = embedding2.flatten()
        
        # Calculate norms
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Normalize and calculate similarity
        normalized1 = emb1 / norm1
        normalized2 = emb2 / norm2
        
        similarity = np.dot(normalized1, normalized2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "input_size": self.input_size,
            "available": self.is_available(),
            "embedding_dimension": 512 if self.is_available() else 0,
            "torch_available": _TORCH_AVAILABLE,
            "facenet_available": _FACENET_AVAILABLE
        }


def create_face_embedding_extractor(model_name: str = "vggface2", 
                                   device: Optional[str] = None,
                                   weights_path: Optional[str] = None,
                                   face_embedding_weights_path: Optional[str] = None) -> FaceEmbeddingExtractor:
    """
    Create a face embedding extractor with specified configuration.
    
    Args:
        model_name: Pre-trained model to use ("vggface2" or "casia-webface")
        device: Device to run inference on (auto-detect if None)
        weights_path: Path to directory containing pre-downloaded weights (to avoid downloading)
        face_embedding_weights_path: Specific path to face embedding weights file (for copying after download)
        
    Returns:
        Configured FaceEmbeddingExtractor instance
    """
    return FaceEmbeddingExtractor(
        model_name=model_name, 
        device=device, 
        weights_path=weights_path,
        face_embedding_weights_path=face_embedding_weights_path
    )
