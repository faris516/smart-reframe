from typing import Tuple, Optional
import numpy as np
from ..tracking.one_euro_filter import OneEuroFilter

class ZoomController:
    def __init__(self, base_height: int, aspect_ratio: float, min_zoom: float = 1.0, max_zoom: float = 2.0):
        """
        Manages the dynamic zoom (crop size) to keep subjects in frame.
        
        Args:
            base_height: The default crop height (e.g., 1080 for 9:16 out of 1080p source).
            aspect_ratio: Width / Height (e.g., 9/16).
            min_zoom: Minimum zoom factor (1.0 = base_height).
            max_zoom: Maximum zoom out factor (e.g., 2.0x base size).
        """
        self.base_height = base_height
        self.aspect_ratio = aspect_ratio
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        
        # Smoother for the zoom factor (very slow beta for cinematic feel)
        self.zoom_filter = OneEuroFilter(min_cutoff=0.01, beta=0.001, d_cutoff=1.0)
        self.pulse = 0.0
        
    def calculate_zoom(self, subject_box: Optional[Tuple[int, int, int, int]], focus_point: Tuple[int, int], source_dims: Tuple[int, int], timestamp: float, audio_energy: float = 0.0) -> float:
        """
        Calculates the required zoom factor with audio-reactive pulse.
        """
        target_zoom = self.min_zoom
        
        # Audio Pulse Logic (The "Breathing" Effect)
        # Decay pulse
        self.pulse *= 0.85 # Fast decay
        
        # Trigger pulse on beats/loud audio
        if audio_energy > 0.7:
             # Add Kick
             self.pulse = min(self.pulse + 0.02, 0.08) # Cap at 8% extra zoom
        
        if subject_box:
            sx, sy, sw, sh = subject_box
            cx, cy = focus_point
            
            # Current crop dimensions at base zoom
            base_crop_h = self.base_height * self.min_zoom
            base_crop_w = base_crop_h * self.aspect_ratio
            
            # Margins (Breathing Room)
            # Optimized for "Efficient Two People" shot
            safe_margin_w = base_crop_w * 0.15 # 15% side margin
            safe_margin_top = base_crop_h * 0.20 # 20% headroom (prevent chopped heads)
            safe_margin_bottom = base_crop_h * 0.10 # 10% footroom
            
            # Distances from center to subject edges
            dist_left = cx - sx
            dist_right = (sx + sw) - cx
            dist_top = cy - sy
            dist_bottom = (sy + sh) - cy
            
            # Required half-width to fit this distance + margin
            req_half_w = max(dist_left, dist_right) + safe_margin_w
            
            # Required half-height (Asymmetric)
            req_half_h_top = dist_top + safe_margin_top
            req_half_h_bottom = dist_bottom + safe_margin_bottom
            req_half_h = max(req_half_h_top, req_half_h_bottom)
            
            # Required full dims
            req_w = req_half_w * 2
            req_h = req_half_h * 2
            
            # Zoom factors needed
            zoom_w = req_w / base_crop_w
            zoom_h = req_h / base_crop_h
            
            # --- CLOSE-UP CORRECTION ---
            # If the face/subject height is very large relative to the crop, force zoom out
            # We want the face to occupy at most 55% of the vertical space to avoid "choked" framing
            subject_h = sh
            max_face_occupancy = 0.55
            
            # Calculate what zoom (multiplier) is needed to make subject_h equal 55% of the new height
            # new_h = base_crop_h * factor
            # subject_h / new_h <= max_face_occupancy
            # subject_h / (base_crop_h * factor) <= max_face_occupancy
            # factor >= subject_h / (base_crop_h * max_face_occupancy)
            
            needed_zoom_for_size = subject_h / (base_crop_h * max_face_occupancy)
            
            # Use the larger requirement (Margins vs Size)
            needed_zoom = max(zoom_w, zoom_h, needed_zoom_for_size)
            
            # Clamp to limits
            target_zoom = max(self.min_zoom, min(needed_zoom, self.max_zoom))
            
            # Safety: Ensure zoomed crop doesn't exceed source video dimensions
            src_w, src_h = source_dims
            max_h_possible = src_h
            max_w_possible = src_w / self.aspect_ratio
            
            abs_h = self.base_height * target_zoom
            abs_w = abs_h * self.aspect_ratio
            
            if abs_h > src_h or abs_w > src_w:
                # Scale back if exceeding source
                scale_h = src_h / abs_h
                scale_w = src_w / abs_w
                scale = min(scale_h, scale_w)
                target_zoom *= scale
        
        # Add Pulse (Viral Energy)
        # Apply pulse AFTER clamping logic so we don't break bounds?
        # Ideally before clamping, but pulse is small.
        # Let's apply it before filter so it smooths out but keeps the punch.
        target_zoom += self.pulse
        
        # Asymmetric Smoothing Logic
        # If we need to zoom OUT (target > current), do it fast to avoid cutting faces.
        # If we need to zoom IN (target < current), do it slow for cinematic feel.
        
        current_z = self.current_zoom if hasattr(self, 'current_zoom') else self.min_zoom
        
        if target_zoom > current_z:
            # Fast expansion
            self.zoom_filter.beta = 0.15 
            self.zoom_filter.min_cutoff = 0.5
        else:
            # Slow contraction
            self.zoom_filter.beta = 0.002
            self.zoom_filter.min_cutoff = 0.05

        # Smooth the zoom
        smoothed_zoom = self.zoom_filter.filter(target_zoom, timestamp)
        self.current_zoom = smoothed_zoom
        
        return smoothed_zoom
