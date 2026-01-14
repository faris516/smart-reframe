import librosa
import numpy as np
import logging
from typing import List

class BeatDetector:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.beats = []
        self.duration = 0.0
        
        try:
            # Load audio (efficiently)
            y, sr = librosa.load(audio_path, sr=22050)
            self.duration = librosa.get_duration(y=y, sr=sr)
            
            # Detect Tempo and Beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            self.beats = librosa.frames_to_time(beat_frames, sr=sr)
            
            logging.info(f"Analyzed Beats: {len(self.beats)} detected. Tempo: {tempo:.2f} BPM")
        except Exception as e:
            logging.error(f"Beat detection failed: {e}")
            self.beats = []

    def get_closest_beat(self, timestamp: float, threshold: float = 0.5) -> float:
        """
        Finds the closest beat to the given timestamp.
        If no beat is close (within threshold), returns the original timestamp.
        """
        if len(self.beats) == 0:
            return timestamp
            
        # Find closest
        idx = (np.abs(self.beats - timestamp)).argmin()
        closest = self.beats[idx]
        
        if abs(closest - timestamp) <= threshold:
            return closest
        
        return timestamp

    def get_beats_in_range(self, start: float, end: float) -> List[float]:
        """Returns all beat timestamps within range."""
        return [b for b in self.beats if start <= b <= end]
