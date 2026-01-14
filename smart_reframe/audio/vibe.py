import librosa
import numpy as np
import logging

class VibeAnalyzer:
    def __init__(self):
        pass

    def analyze(self, audio_path):
        """Returns (bpm, energy) tuple."""
        try:
            # Load audio (only first 60s to save time)
            y, sr = librosa.load(audio_path, duration=60)
            
            # 1. BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)
            
            # 2. Energy (RMS)
            rms = librosa.feature.rms(y=y)
            energy = float(np.mean(rms))
            
            logging.info(f"Vibe Analysis: BPM={bpm:.1f}, Energy={energy:.3f}")
            return bpm, energy
            
        except Exception as e:
            logging.error(f"Vibe analysis failed: {e}")
            return 0.0, 0.0
