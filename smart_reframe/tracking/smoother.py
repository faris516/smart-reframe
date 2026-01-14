import numpy as np
from typing import Tuple, Optional
from .one_euro_filter import OneEuroFilter

class Stabilizer:
    def __init__(self, min_cutoff=0.1, beta=0.01, d_cutoff=1.0, dead_zone_px=10.0, max_vel_px=30.0):
        """
        Stabilizer using One Euro Filter and Dead Zone with Velocity Clamping.
        
        Args:
            min_cutoff: Min cutoff frequency (Hz). Lower = more smoothing.
            beta: Speed coefficient. Higher = less lag.
            dead_zone_px: Movement below this threshold (pixels) is ignored.
            max_vel_px: Maximum allowed movement per update (pixels) for gradual panning.
        """
        self.x_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.dead_zone_px = dead_zone_px
        self.max_vel_px = max_vel_px
        
        self.last_clean_pos = None # Last raw position that triggered an update
        self.last_smoothed_pos = None

    def update(self, current_val: Tuple[float, float], timestamp: float) -> Tuple[float, float]:
        """
        Updates the smoothed value.
        Args:
            current_val: (x, y) tuple.
            timestamp: Current timestamp in seconds.
        """
        x, y = current_val
        
        # Dead Zone Logic
        if self.last_clean_pos is not None:
             dist = np.sqrt((x - self.last_clean_pos[0])**2 + (y - self.last_clean_pos[1])**2)
             if dist < self.dead_zone_px:
                 if self.last_smoothed_pos is not None:
                     return self.last_smoothed_pos
        
        self.last_clean_pos = (x, y)
        
        sx = self.x_filter.filter(x, timestamp)
        sy = self.y_filter.filter(y, timestamp)
        
        # Velocity Clamping (Gradual Switching)
        if self.max_vel_px > 0 and self.last_smoothed_pos is not None:
            lx, ly = self.last_smoothed_pos
            dx = sx - lx
            dy = sy - ly
            dist_move = np.sqrt(dx*dx + dy*dy)
            
            if dist_move > self.max_vel_px:
                scale = self.max_vel_px / dist_move
                sx = lx + dx * scale
                sy = ly + dy * scale
                # Note: We don't update the internal OneEuro filter state filter value here
                # because that would feedback the clamped value. 
                # Ideally we want the filter to track the real target but the OUTPUT to be clamped.
                # But next frame filter starts from its own internal state. 
                # Effectively this just lags the output.
        
        self.last_smoothed_pos = (sx, sy)
        return (sx, sy)

class CinematicStabilizer:
    def __init__(self, min_cutoff=0.04, beta=0.004, d_cutoff=1.0, dead_zone_ratio=0.10, max_vel_px=15.0):
        """
        Advanced Stabilizer simulating a professional cameraman "Virtual Tripod".
        Uses a "Safe Box" (hysteresis): The camera does not move if the subject is within the box.
        If the subject pushes the edge, the camera moves to re-center them.
        
        Args:
            dead_zone_ratio: Fraction of frame width/height that is the "safe zone" (0.10 = 10%).
        """
        self.x_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        
        self.dead_zone_ratio = dead_zone_ratio
        self.max_vel_px = max_vel_px
        
        # State
        self.cam_pos = None # The virtual camera center
        self.frame_dims = (1920, 1080) # Default, will be updated on first run
        
    def reset(self, initial_pos: Optional[Tuple[float, float]] = None):
        self.x_filter.reset()
        self.y_filter.reset()
        self.cam_pos = initial_pos

    def update(self, current_val: Tuple[float, float], timestamp: float, frame_dims: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        if frame_dims:
            self.frame_dims = frame_dims
            
        width, height = self.frame_dims
        margin_x = width * self.dead_zone_ratio
        margin_y = height * self.dead_zone_ratio * 0.6 
        
        raw_x, raw_y = current_val
        
        if self.cam_pos is None:
            self.cam_pos = (raw_x, raw_y)
            return (raw_x, raw_y)
            
        curr_cx, curr_cy = self.cam_pos
        
        # --- 1. Tripod/Drag Logic ---
        # Calculate target camera position
        # Refined Logic: If outside deadzone, target the TRUE CENTER (raw_x), not just the edge.
        # This ensures we drift back to center during movement, rather than maintaining an offset.
        
        # X-Axis
        diff_x = raw_x - curr_cx
        target_x = curr_cx
        
        if abs(diff_x) > margin_x:
            # Subject outside safe zone -> Re-center!
            # We target raw_x (the subject). The OneEuro filter will smooth the approach.
            target_x = raw_x
                
        # Y-Axis
        diff_y = raw_y - curr_cy
        target_y = curr_cy
        
        if abs(diff_y) > margin_y:
            target_y = raw_y
        
        # --- 2. Smoothing (OneEuro) ---
        # Smooth the movement from current cam_pos to target
        # Note: OneEuro creates momentum. 
        sx = self.x_filter.filter(target_x, timestamp)
        sy = self.y_filter.filter(target_y, timestamp)
        
        # --- 3. Velocity Clamping ---
        dx = sx - curr_cx
        dy = sy - curr_cy
        dist_move = np.sqrt(dx*dx + dy*dy)
        if self.max_vel_px > 0 and dist_move > self.max_vel_px:
             scale = self.max_vel_px / dist_move
             sx = curr_cx + dx * scale
             sy = curr_cy + dy * scale
             
        self.cam_pos = (sx, sy)
        return (sx, sy)

class EMASmoother:
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.last_val = None

    def update(self, current_val: Tuple[float, float], timestamp: Optional[float] = None) -> Tuple[float, float]:
        # Timestamp is ignored for simple EMA, but kept for interface compatibility
        if self.last_val is None:
            self.last_val = current_val
            return current_val
        
        cx, cy = current_val
        lx, ly = self.last_val
        
        nx = self.alpha * lx + (1 - self.alpha) * cx
        ny = self.alpha * ly + (1 - self.alpha) * cy
        
        self.last_val = (nx, ny)
        return (nx, ny)

    def reset(self, initial_pos: Optional[Tuple[float, float]] = None):
        self.last_val = initial_pos
