# smart_reframe/config.py

# Video Configuration
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
TARGET_ASPECT_RATIO = 9 / 16
MIN_FPS = 10

# Tracking Configuration
SMOOTHING_ALPHA = 0.85  # EMA factor: higher = more smoothing, less responsiveness
# One Euro Filter parameters
# min_cutoff: Lower = smoother (less jitter) at low speeds. 0.05 is very smooth.
# beta: Higher = more responsive (less lag) at high speeds. 0.005 is balanced.
OE_MIN_CUTOFF = 0.05
OE_BETA = 0.005
OE_D_CUTOFF = 1.0
DEAD_ZONE_PX = 15  # Ignore movements smaller than this (pixels)
MAX_PAN_SPEED = 20 # Max pixels per frame for gradual panning (approx 600px/s at 30fps)
DETECTION_INTERVAL = 5  # Run expensive detectors every N frames
MAX_GROUP_DISTANCE_RATIO = 0.5 # Max distance between subjects (relative to frame width) to group them
MIN_GROUP_CONFIDENCE = 0.6 # Minimum combined confidence to form a group

# Split Screen Configuration
SPLIT_SCREEN_THRESHOLD = 0.5 # If distance > this * width, trigger split screen
SPLIT_BORDER_THICKNESS = 10
SPLIT_BORDER_COLOR = (0, 0, 0) # Black border

# Audio / VAD Configuration
VAD_THRESHOLD = 0.01    # RMS energy threshold for speech detection
VAD_WINDOW_SECONDS = 0.5

# Scene Detection
SCENE_CUT_THRESHOLD = 0.5 # Increased to avoid false positives (was 0.3)

# Priority Weights (Speaker Scoring)
WEIGHT_AUDIO_ACTIVITY = 2.0
WEIGHT_MOUTH_ENERGY = 1.5
WEIGHT_FACE_CONFIDENCE = 1.0
WEIGHT_CENTER_BIAS = 0.5 # Penalty for faces far from center

# Shorts Automator Configuration
SHORTS_SEGMENT_DURATION = 5.0 # Duration of each analysis chunk (seconds)
SHORTS_TARGET_DURATION = 58.0 # Target total length (allowing 2s buffer for <60s)
SHORTS_TEXT_EDGE_THRESHOLD = 0.15 # Edge density threshold (Top/Bottom)
SHORTS_TEXT_CENTER_THRESHOLD = 0.30 # Stricter threshold for Center text (Titles/Standalone)
SHORTS_MIN_SEGMENT_SCORE = 0.2 # Minimum score to consider a segment usable

# Paths
DEFAULT_OUTPUT_FILENAME = "output.mp4"
