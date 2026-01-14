import cv2
import mediapipe as mp
import numpy as np
import os
import logging

class FaceDetector:
    def __init__(self):
        self.use_tasks_api = False
        self.detector = None
        
        # Try Loading Legacy Solutions API
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            logging.info("Using MediaPipe Solutions API")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5, 
                refine_landmarks=True, 
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Fallback to Tasks API
            logging.info("Using MediaPipe Tasks API")
            self.use_tasks_api = True
            
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')
            if not os.path.exists(model_path):
                # Try absolute path based on CWD if relative fails
                model_path = 'smart_reframe/models/face_landmarker.task'
                
            if not os.path.exists(model_path):
                logging.error(f"Face Landmarker model not found at {model_path}. Face detection will vary.")
                return

            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self.detector = FaceLandmarker.create_from_options(options)
            self.last_timestamp_ms = 0

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> list:
        """
        Detects faces in the frame.
        Args:
            frame: BGR numpy array
            timestamp: Timestamp in seconds (required for Tasks API)
        Returns a list of dicts: {'bbox': (x, y, w, h), 'landmarks': [...]}
        """
        if self.detector is None:
            return []
            
        h, w = frame.shape[:2]
        
        # Downscale for performance (Max width 720)
        target_w = 720
        if w > target_w:
            scale = target_w / w
            small_w = target_w
            small_h = int(h * scale)
            processing_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            processing_frame = frame
            
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        faces = []
        if not self.use_tasks_api:
            # Legacy Solutions API
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # landmarks.landmark is a RepeatedCompositeContainer
                    lms = landmarks.landmark
                    
                    # Landmarks are normalized (0-1). 
                    # They apply to the original image since we preserved aspect ratio.
                    # We usually multiply by W/H to get pixels. 
                    # We should multiply by ORIGINAL W, H.
                    
                    xs = [lm.x for lm in lms]
                    ys = [lm.y for lm in lms]
                    x1, x2 = min(xs) * w, max(xs) * w
                    y1, y2 = min(ys) * h, max(ys) * h
                    
                    faces.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'landmarks': lms, 
                        'confidence': 1.0
                    })
        else:
            # Tasks API
            # Use provided timestamp or fallback to monotonic counter if 0
            # Tasks API requires monotonically increasing timestamps
            if timestamp > 0:
                current_timestamp_ms = int(timestamp * 1000)
            else:
                current_timestamp_ms = int(time.time() * 1000)

            # Ensure strict monotonicity just in case
            if current_timestamp_ms <= self.last_timestamp_ms:
                current_timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = current_timestamp_ms
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect_for_video(mp_image, current_timestamp_ms)
            
            if results.face_landmarks:
                for landmarks_list in results.face_landmarks:
                    # landmarks_list is a list of NormalizedLandmark
                    # Compatible with .x, .y access
                    
                    # Same logic: Normalized coords apply to original frame
                    xs = [lm.x for lm in landmarks_list]
                    ys = [lm.y for lm in landmarks_list]
                    x1, x2 = min(xs) * w, max(xs) * w
                    y1, y2 = min(ys) * h, max(ys) * h
                    
                    # Calculate "Frontality" (Facing Score)
                    # Using nose tip (1) relative to cheekbones/ears
                    # Simpler robust metric: Symmetry of nose between eyes
                    # Left Eye Inner: 133, Right Eye Inner: 362, Nose Tip: 1
                    # Note: MediaPipe indices are fixed.
                    try:
                        nose = landmarks_list[1]
                        left_eye = landmarks_list[133] # Inner corner
                        right_eye = landmarks_list[362] # Inner corner
                        
                        eye_dist = right_eye.x - left_eye.x
                        nose_offset = nose.x - left_eye.x
                        
                        if eye_dist > 0:
                            ratio = nose_offset / eye_dist
                            # Ideal center is 0.5
                            # Deviation from 0.5 implies turning head
                            deviation = abs(ratio - 0.5)
                            # Max expected deviation ~0.5 (nose passes eye)
                            # Score 1.0 at 0 deviation, 0.0 at 0.5 deviation
                            facing_score = max(0.0, 1.0 - (deviation * 2.5)) # Steep dropoff
                        else:
                            facing_score = 0.5 # Unknown
                    except IndexError:
                        facing_score = 1.0
                    
                    faces.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'landmarks': landmarks_list,
                        'confidence': 1.0, # Tasks API doesn't give per-face confidence easily here
                        'facing_score': facing_score
                    })

        return faces

    def close(self):
        if self.detector:
            self.detector.close()

import time # Needed for Tasks API timestamp
