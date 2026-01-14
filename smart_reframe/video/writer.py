import subprocess
import logging
import numpy as np
from typing import Optional

class VideoWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float, source_path: Optional[str] = None, start_time: float = 0.0, duration: Optional[float] = None):
        """
        Initializes the video writer.
        
        Args:
            output_path: Path to save the output video.
            width: Output width.
            height: Output height.
            fps: Output frames per second.
            source_path: Path to the source video (for audio mapping). If None, no audio is written.
            start_time: Start time in seconds for audio trimming.
            duration: Duration in seconds for audio trimming.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',                  # Overwrite output file
            '-f', 'rawvideo',      # Input format
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', # Input resolution
            '-pix_fmt', 'bgr24',   # OpenCV uses BGR
            '-r', str(fps),        # Input framerate
            '-i', '-',             # Read from stdin
        ]
        
        if source_path:
            # For audio trimming, we must apply -ss BEFORE -i
            audio_args = ['-ss', str(start_time)]
            if duration is not None:
                audio_args.extend(['-t', str(duration)])
                
            cmd.extend(audio_args)
            cmd.extend([
                '-i', source_path,     # Input audio source (usually same as video input)
                '-map', '0:v',         # Use video from input 0 (stdin)
                '-map', '1:a?',        # Use audio from input 1 (source), ? makes it optional if no audio track
                '-c:a', 'aac',         # Audio codec
                '-b:a', '192k',        # Audio bitrate
            ])
            
        cmd.extend([
            # Hardware Acceleration Check (macOS)
            # Use VideoToolbox if available for massive speedup
            '-c:v', 'h264_videotoolbox', 
            '-b:v', '6000k',        # 6Mbps is sufficient for vertical 1080p
            '-allow_sw', '1',      # Allow software fallback if HW fails
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-shortest',
            output_path
        ])
        
        logging.info(f"FFmpeg command: {' '.join(cmd)}")
        
        # Start FFmpeg process
        try:
            self.process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL # Avoid deadlock by ignoring stderr
            )
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please ensure FFmpeg is installed and in your PATH.")

    def write(self, frame: np.ndarray):
        """Writes a frame to the video stream."""
        if self.process is None:
            raise RuntimeError("VideoWriter is closed.")
            
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
             raise ValueError(f"Frame shape {frame.shape} does not match writer resolution {self.width}x{self.height}")

        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            stderr = self.process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg process crashed: {stderr}")

    def release(self):
        """Closes the pipe and waits for FFmpeg to finish."""
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait()
            self.process = None
            logging.info("VideoWriter released.")
