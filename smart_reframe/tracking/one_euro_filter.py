import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        One Euro Filter for adaptive smoothing.
        
        Args:
            min_cutoff: Min cutoff frequency (Hz). Lower = more smoothing for slow movements.
            beta: Speed coefficient. Higher = more responsiveness to fast movements.
            d_cutoff: Cutoff frequency for derivative (Hz).
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None
        
    def filter(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        # First sample
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter_val(x, alpha=1.0) # No smoothing
            
        dt = timestamp - self.last_time
        self.last_time = timestamp
        
        # Avoid division by zero
        if dt <= 0:
            return self.x_filter.last_val
            
        # Estimate derivative (speed)
        dx = (x - self.x_filter.last_val) / dt
        edx = self.dx_filter.filter_val(dx, alpha=self._alpha(dt, self.d_cutoff))
        
        # Use speed to calculate dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Filter signal
        return self.x_filter.filter_val(x, alpha=self._alpha(dt, cutoff))
        
    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

class LowPassFilter:
    def __init__(self):
        self.last_val = None
        
    def filter_val(self, val, alpha):
        if self.last_val is None:
            s = val
        else:
            s = alpha * val + (1.0 - alpha) * self.last_val
        self.last_val = s
        return s
        
    def reset(self):
        self.last_val = None
