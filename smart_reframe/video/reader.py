import cv2
import os
from typing import Generator, Tuple, Optional
import logging

class VideoReader:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        
        self.path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logging.info(f"Opened video: {path} ({self.width}x{self.height} @ {self.fps:.2f}fps, {self.duration:.2f}s)")

    def frames(self) -> Generator[Tuple[bool, Optional[object]], None, None]:
        """Yields frames from the video (legacy tuple format)."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield ret, frame

    def __iter__(self):
        """Allows direct iteration over frames (yields frame image only)."""
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame
    
    def set_position(self, timestamp: float):
        """Seek to a specific timestamp in seconds."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        
    def get_timestamp(self) -> float:
        """Returns current timestamp in seconds."""
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
    def release(self):
        self.cap.release()
