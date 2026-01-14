import os
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile
import logging
from ..config import VAD_THRESHOLD

class AudioAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.sample_rate = 16000
        self.audio_data = None
        self.duration = 0
        
        self._extract_and_load_audio()

    def _extract_and_load_audio(self):
        """Extracts audio to a temporary WAV file and loads it."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_path = tmp_audio.name
            
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', self.video_path,
                '-vn',              # No video
                '-ac', '1',         # Mono
                '-ar', str(self.sample_rate),
                '-f', 'wav',
                tmp_path
            ]
            
            # Run ffmpeg (capture output to avoid clutter)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Read WAV
            rate, data = wavfile.read(tmp_path)
            
            # Normalize to -1..1
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                 data = data.astype(np.float32) / 2147483648.0
                 
            self.audio_data = data
            self.duration = len(data) / rate
            logging.info(f"Audio extracted: {self.duration:.2f}s, {len(data)} samples.")
            
        except subprocess.CalledProcessError:
            logging.warning("Failed to extract audio. VAD will be disabled.")
            self.audio_data = np.array([])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_audio_energy(self, start_time: float, end_time: float) -> float:
        """Returns the RMS energy of the audio segment."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
            
        start_idx = int(start_time * self.sample_rate)
        end_idx = int(end_time * self.sample_rate)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(self.audio_data), end_idx)
        
        if start_idx >= end_idx:
            return 0.0
            
        segment = self.audio_data[start_idx:end_idx]
        rms = np.sqrt(np.mean(segment**2))
        return float(rms)

    def is_active(self, timestamp: float, window: float = 0.5) -> bool:
        """Simple threshold-based VAD."""
        start = max(0, timestamp - window / 2)
        end = timestamp + window / 2
        energy = self.get_audio_energy(start, end)
        return energy > VAD_THRESHOLD
