import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .reid import FaceReID

class KalmanTrack:
    """
    Represents a single tracked object with a Kalman Filter.
    State: [x, y, dx, dy] (Center X, Center Y, Velocity X, Velocity Y)
    Measurement: [x, y]
    """
    def __init__(self, track_id: int, initial_bbox: Tuple[int, int, int, int], embedding: Optional[np.ndarray] = None):
        self.track_id = track_id
        self.embedding = embedding
        self.embedding_samples = 1 if embedding is not None else 0
        
        # Initialize Kalman Filter
        # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition Matrix (Physics Model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (We observe x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Noise Covariance Matrices
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement Noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1 
        
        # Error Covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial State
        cx = initial_bbox[0] + initial_bbox[2] / 2
        cy = initial_bbox[1] + initial_bbox[3] / 2
        self.kf.statePost = np.array([cx, cy, 0, 0], np.float32)
        
        # Formatting State
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.last_bbox = initial_bbox
        self.frames_consecutive_hit = 1

    def predict(self):
        """Advances the state vector using the transition matrix."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.state()

    def update(self, bbox: Tuple[int, int, int, int], embedding: Optional[np.ndarray] = None):
        """Updates the state vector with observed measurement."""
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        
        self.last_bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.frames_consecutive_hit += 1
        
        # Update Embedding (EMA-like)
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                # Weighted average to adapt slowly to angle changes
                alpha = 0.1
                self.embedding = (1 - alpha) * self.embedding + alpha * embedding
                self.embedding = self.embedding / np.linalg.norm(self.embedding) # Renormalize

    def state(self):
        """Returns the current predicted (cx, cy)."""
        return (self.kf.statePost[0], self.kf.statePost[1])
    
    def get_bbox(self):
        """Returns the estimated full bbox (using last known size)."""
        cx, cy = self.state()
        w, h = self.last_bbox[2], self.last_bbox[3]
        x = int(cx - w/2)
        y = int(cy - h/2)
        return (x, y, w, h)


class MultiObjectTracker:
    def __init__(self, max_age: int = 60, max_dist_ratio: float = 0.2, reid_threshold: float = 0.4):
        self.tracks: List[KalmanTrack] = []
        self.lost_tracks: List[KalmanTrack] = [] # Memory of lost tracks for Re-ID
        self.next_id = 0
        self.max_age = max_age 
        self.max_dist_ratio = max_dist_ratio 
        self.reid_threshold = reid_threshold
        
        self.reid = FaceReID() # Initialize Re-ID model

    def update(self, detections: List[dict], frame_width: int, frame_image: Optional[np.ndarray] = None):
        """
        updates tracks with new detections.
        """
        # 1. Predict new locations for existing tracks
        for t in self.tracks:
            t.predict()
            
        # 2. Extract Embeddings if image available
        # Modifies detections in-place
        if frame_image is not None and self.reid.recognizer is not None:
            for det in detections:
                if 'embedding' not in det and 'landmarks' in det:
                    det['embedding'] = self.reid.extract_embedding(frame_image, det['landmarks'])

        # 3. Match detections to LIVE tracks (Spatial)
        unmatched_dets = []
        
        det_centers = []
        for d in detections:
            b = d['bbox']
            det_centers.append((b[0] + b[2]/2, b[1] + b[3]/2))
            
        used_tracks = set()
        used_dets = set()
        
        threshold = frame_width * self.max_dist_ratio
        
        candidates = []
        for t_idx, track in enumerate(self.tracks):
             t_pos = track.state()
             for d_idx, d_pos in enumerate(det_centers):
                 dist = np.sqrt((t_pos[0]-d_pos[0])**2 + (t_pos[1]-d_pos[1])**2)
                 if dist < threshold:
                     candidates.append((dist, t_idx, d_idx))
                     
        # Sort by lowest distance
        candidates.sort(key=lambda x: x[0])
        
        for dist, t_idx, d_idx in candidates:
            if t_idx not in used_tracks and d_idx not in used_dets:
                # Match found!
                self.tracks[t_idx].update(detections[d_idx]['bbox'], detections[d_idx].get('embedding'))
                
                detections[d_idx]['track_id'] = self.tracks[t_idx].track_id
                detections[d_idx]['track_count'] = self.tracks[t_idx].age
                
                used_tracks.add(t_idx)
                used_dets.add(d_idx)

        # 4. Handle Unmatched Detections -> Try Re-ID with LOST tracks
        unmatched_det_indices = [i for i in range(len(detections)) if i not in used_dets]
        
        for d_idx in unmatched_det_indices:
            det = detections[d_idx]
            emb = det.get('embedding')
            
            matched_lost_id = None
            if emb is not None:
                # Check against lost tracks
                best_score = self.reid_threshold
                best_match_idx = -1
                
                for l_idx, l_track in enumerate(self.lost_tracks):
                    if l_track.embedding is not None:
                         score = self.reid.compute_similarity(emb, l_track.embedding)
                         # SFace Cosine: Higher is better. 0.363 is threshold.
                         if score > best_score:
                             best_score = score
                             best_match_idx = l_idx
                             
                if best_match_idx != -1:
                    # Resurrect Track!
                    restored_track = self.lost_tracks.pop(best_match_idx)
                    restored_track.kf.statePost = np.array([
                        det['bbox'][0] + det['bbox'][2]/2,
                        det['bbox'][1] + det['bbox'][3]/2,
                        0, 0], np.float32) # Reset velocity/pos
                    
                    restored_track.update(det['bbox'], emb)
                    restored_track.time_since_update = 0 # Mark alive
                    
                    logging.info(f"Re-ID: Resurrected Star #{restored_track.track_id} (Score: {best_score:.3f})")
                    
                    self.tracks.append(restored_track)
                    
                    det['track_id'] = restored_track.track_id
                    det['track_count'] = restored_track.age
                    matched_lost_id = restored_track.track_id

            if matched_lost_id is None:
                # New Track
                self._create_track(det['bbox'], emb)
                det['track_id'] = self.next_id - 1
                det['track_count'] = 1
                
        # 5. Handle Unmatched Tracks (Lost) -> Move to Lost Bin
        active_tracks = []
        for t in self.tracks:
            if t.time_since_update < 5: # Keep alive short term
                active_tracks.append(t)
            else:
                # Move to lost tracks if it has an embedding (worth remembering)
                if t.embedding is not None and t.hits > 15: # Only remember solid tracks
                    # Avoid duplicates in lost tracks - if already there ?
                    # Prune old lost tracks
                    self.lost_tracks.append(t)
                # Else: die
                pass
                
        self.tracks = active_tracks
        
        # Prune Lost Tracks (Time limit)
        # We can keep them longer (e.g. 5 minutes) if they have embeddings
        # SFace is robust. Let's keep 2000 frames (~1 min)
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update < 2000]
        
    def _create_track(self, bbox, embedding=None):
        self.tracks.append(KalmanTrack(self.next_id, bbox, embedding))
        self.next_id += 1
