import numpy as np
from scipy.ndimage import gaussian_filter1d
import logging

class SaliencyEngine:
    """
    Converts discrete detections (Face, Person, Motion) into a continuous 1D importance map.
    The map represents the "value" of each X-coordinate in the frame.
    """
    def __init__(self, width: int):
        self.width = width
        # Weights for different cues
        self.WEIGHT_FACE = 1.0
        self.WEIGHT_PERSON = 0.5
        self.WEIGHT_MOTION = 0.3
        self.WEIGHT_CENTER_BIAS = 0.1
        
        # Sigma for gaussian splatter (smoothness of influence)
        self.SIGMA_FACE = width * 0.05 # 5% of width
        self.SIGMA_PERSON = width * 0.1 # 10% of width
        
        # Grid
        self.grid = np.arange(width)

    def compute_saliency(self, faces: list, people: list, motion_center: tuple = None) -> np.ndarray:
        """
        Generates a normalized 1D saliency map.
        Values range roughly [0, 1] but can go higher with overlaps.
        """
        # Initialize map with small epsilon to avoid zero division
        saliency = np.ones(self.width) * 0.001
        
        # 1. Add Faces (High Importance, Sharp Peaks)
        if faces:
            for face in faces:
                bbox = face['bbox']
                cx = bbox[0] + bbox[2] / 2
                w = bbox[2]
                confidence = face.get('confidence', 1.0)
                
                # Create Gaussian centered at cx
                # Importance roughly proportional to size (larger face = more important?)
                # Actually, equal importance for main faces, but we can weigh by confidence.
                
                # Normalized coordinate
                peak_val = self.WEIGHT_FACE * confidence
                
                # Splat Gaussian
                saliency += self._gaussian(cx, self.SIGMA_FACE, peak_val)

        # 2. Add People (Medium Importance, Broad Peaks)
        if people:
            for person in people:
                bbox = person['bbox']
                cx = bbox[0] + bbox[2] / 2
                confidence = person.get('confidence', 1.0)
                
                peak_val = self.WEIGHT_PERSON * confidence
                saliency += self._gaussian(cx, self.SIGMA_PERSON, peak_val)
                
        # 3. Add Motion (Low Importance, Background context)
        if motion_center:
            mx, my = motion_center
            # Wide gaussian for general movement area
            saliency += self._gaussian(mx, self.width * 0.2, self.WEIGHT_MOTION)
            
        # 4. Center Bias (Cinematography rule: keep action central if unsure)
        # Adds a weak gravity to the center
        # saliency += self._gaussian(self.width / 2, self.width * 0.4, self.WEIGHT_CENTER_BIAS)
        
        # Normalize?
        # Actually, Solver needs absolute density to maximize "captured value".
        # So we keep it additive.
        
        return saliency

    def _gaussian(self, mu, sigma, height):
        return height * np.exp(-0.5 * ((self.grid - mu) / sigma) ** 2)

    def get_window_score(self, saliency: np.ndarray, x_start: int, window_width: int) -> float:
        """Calculates total importance captured by a window."""
        x_end = min(x_start + window_width, self.width)
        start = max(0, int(x_start))
        
        if start >= self.width:
            return 0.0
            
        return np.sum(saliency[start:int(x_end)])
