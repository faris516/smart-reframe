import cv2
import numpy as np
import os
import logging
from typing import List, Optional

class FaceReID:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default location
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_recognition_sface_2021dec.onnx')
            
        self.recognizer = None
        if os.path.exists(model_path):
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model_path, 
                    "" # config path not needed for ONNX
                )
                logging.info(f"Loaded SFace Re-ID model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load SFace model: {e}")
        else:
            logging.warning(f"SFace model not found at {model_path} - Re-ID disabled.")

    def extract_embedding(self, frame: np.ndarray, landmarks: list) -> Optional[np.ndarray]:
        """
        Extracts 128D embedding from a face using 5-point alignment.
        Args:
            frame: BGR image
            landmarks: List of MediaPipe landmarks (NormalizedLandmark)
        """
        if self.recognizer is None or not landmarks:
            return None
            
        h, w = frame.shape[:2]
        
        # Convert MP landmarks to 5-point format required by FaceRecognizerSF
        # MediaPipe 468/478 format indices for eyes, nose, mouth corners:
        # Left Eye: 33, Right Eye: 263, Nose: 1, Left Mouth: 61, Right Mouth: 291
        # SFace expects: right_eye, left_eye, nose, right_mouth, left_mouth (in some order, actually it aligns automatically if decent)
        # Actually cv2.FaceRecognizerSF.alignCrop expects specific formatting.
        # Let's use standard indices for alignment.
        
        idx_map = [468, 473, 1, 61, 291] # L_Eye, R_Eye, Nose, L_Lip, R_Lip (MediaPipe Iris/Mesh indices)
        # Fallback if Iris not present: 33, 263
        
        # Check standard mesh size
        if len(landmarks) > 468:
             key_indices = [468, 473, 1, 61, 291] # Iris centers if available
        else:
             key_indices = [33, 263, 1, 61, 291] # Eye corners
             
        points = []
        for idx in key_indices:
            lm = landmarks[idx]
            points.append([float(lm.x * w), float(lm.y * h)])
            
        points = np.array(points, dtype=np.float32).reshape((5, 2))
        
        # Align and Crop
        # SFace aligns to 112x112
        aligned_face = self.recognizer.alignCrop(frame, points)
        
        if aligned_face is None:
            return None
            
        # Extract Feature
        feat = self.recognizer.feature(aligned_face)
        return feat[0] # Return flat array (128,)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes Cosine Similarity.
        SFace outputs normalized vectors, so dot product is cosine similarity.
        Range: [-1, 1], Threshold commonly around 0.363 for SFace.
        """
        if emb1 is None or emb2 is None:
            return 0.0
            
        score = self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)
        return score
