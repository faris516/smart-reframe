import cv2
import numpy as np
import logging
from tqdm import tqdm
import time
import math

from .video.reader import VideoReader
from .video.writer import VideoWriter
from .audio.speaker_activity import AudioAnalyzer
from .detectors.face_detector import FaceDetector
from .detectors.person_detector import PersonDetector
from .detectors.motion_detector import MotionDetector
from .detectors.mouth_motion import MouthMotionDetector
from .detectors.text_detector import TextDetector
from .tracking.smoother import Stabilizer, EMASmoother, CinematicStabilizer
from .tracking.tracker import MultiObjectTracker
from .framing.cropper import FocusSelector, DynamicCropper
from .solver import SaliencyEngine, PathSolver
from .framing.zoom import ZoomController
from .framing.group import GroupSelector
from .compositor.split_screen import SplitScreenCompositor
from .config import *

class ReframePipeline:
    def __init__(self, input_path: str, output_path: str, output_width: int = OUTPUT_WIDTH, output_height: int = OUTPUT_HEIGHT, stabilizer_type: str = "one_euro", dynamic_zoom: bool = False, group_aware: bool = False, split_screen: bool = False, couple_zoom: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.output_width = output_width
        self.output_height = output_height
        self.couple_zoom_mode = couple_zoom
        
        # Initialize components
        self.reader = VideoReader(input_path)
        self.writer = VideoWriter(
            output_path, 
            self.output_width, 
            self.output_height, 
            self.reader.fps, 
            source_path=input_path
        )
        self.audio = AudioAnalyzer(input_path)
        
        self.face_detector = FaceDetector()
        self.person_detector = PersonDetector()
        self.motion_detector = MotionDetector()
        self.mouth_detector = MouthMotionDetector()
        self.text_detector = TextDetector()
        
        if stabilizer_type == "ema":
             self.smoother = EMASmoother(alpha=SMOOTHING_ALPHA)
        else:
             # "one_euro" now defaults to Cinematic Tripod logic
             self.smoother = CinematicStabilizer(
                min_cutoff=OE_MIN_CUTOFF,
                beta=OE_BETA,
                d_cutoff=OE_D_CUTOFF,
                dead_zone_ratio=0.10, # 10% safety margin (Tighter for centering)
                max_vel_px=MAX_PAN_SPEED
            )
        self.focus_selector = FocusSelector()
        
        target_ratio = self.output_width / self.output_height
        
        # Calculate base crop height constrained by source dimensions
        limit_h_by_width = self.reader.width / target_ratio
        self.base_crop_height = int(min(self.reader.height, limit_h_by_width))
        
        self.cropper = DynamicCropper(
            self.reader.width, 
            self.reader.height, 
            aspect_ratio=target_ratio
        )
        self.cropper.output_height = self.base_crop_height
        
        self.zoom_controller = None
        if dynamic_zoom:
             self.zoom_controller = ZoomController(
                 base_height=self.base_crop_height,
                 aspect_ratio=target_ratio,
                 min_zoom=1.0,
                 max_zoom=1.5
             )
             
        self.group_selector = None
        if group_aware:
            self.group_selector = GroupSelector()
            
        self.compositor = None
        if split_screen:
            self.compositor = SplitScreenCompositor(output_width, output_height)
        
        # State
        self.frame_count = 0
        self.is_split_active = False
        self.split_targets = []
        
        self.last_faces = []
        self.last_motion = None
        self.prev_hist = None
        
        # Main Character Tracking (Persistence)
        self.tracker = MultiObjectTracker(max_age=60, max_dist_ratio=0.2)
        
        # Focus Robustness (Anti-Flicker)
        self.last_valid_focus = None
        self.focus_loss_frames = 0
        self.MAX_FOCUS_LOSS = 30 
        
        # Star Locking (Phase 12)
        self.locked_star_id = None
        self.lock_unlock_timer = 0
        self.LOCK_PERSISTENCE_REQ = 10 
        self.locked_star_id = None
        self.lock_unlock_timer = 0
        self.LOCK_PERSISTENCE_REQ = 10 
        self.UNLOCK_TIMEOUT = 60 
        
        self.last_group_info = None 
        
        # Cinema Mode State (Phase 15)
        self.cinema_mode_active = False
        self.cinema_hold_timer = 0
        self.CINEMA_HOLD_MAX = 30 # Hold for 1 second (at 30fps)
        self.CINEMA_TRIGGER_COUNT = 0
        self.CINEMA_TRIGGER_REQ = 5 # Need 5 frames of wide group to switch 
        self.cinema_transition = 0.0 # 0.0 = Vertical, 1.0 = Cinema
        self.TRANSITION_SPEED = 0.05 # 20 frames to switch
        
        # Solver Components (Phase 11)
        self.saliency_engine = SaliencyEngine(self.reader.width)
        crop_width = int(self.base_crop_height * target_ratio)
        self.path_solver = PathSolver(self.reader.width, crop_width)
        self.solver_mode = False # Default back to False (User feedback: Solver quality poor)
        
        # User Overrides (Phase 16)
        self.user_star_ids = []
        self.user_ignored_ids = []
        self.manual_keyframes = [] # List of {'time': float, 'x': int, 'y': int}
        
        # Speaker Refinement State (Phase A)
        self.last_speaker_idx = None
        self.speaker_streak = 0
        self.SPEAKER_SWITCH_THRESH = 1.2 # New speaker must be 20% better to switch immediately
        self.SPEAKER_MIN_HOLD = 15 # Hold speaker for 0.5s minimum

    def set_user_overrides(self, stars=None, ignored=None, keyframes=None):
        if stars: self.user_star_ids = stars
        if ignored: self.user_ignored_ids = ignored
        if keyframes: 
            self.manual_keyframes = sorted(keyframes, key=lambda k: k['time'])
        logging.info(f"Overrides: Stars={len(self.user_star_ids)}, Ignored={len(self.user_ignored_ids)}, Keys={len(self.manual_keyframes)}")
        
    def _get_manual_focus(self, timestamp):
        """Linearly interpolates between keyframes."""
        if not self.manual_keyframes:
            return None
            
        # Find surrounding keys
        prev_k = None
        next_k = None
        
        for k in self.manual_keyframes:
            if k['time'] <= timestamp:
                prev_k = k
            else:
                next_k = k
                break
                
        if prev_k and not next_k:
            return (prev_k['x'], prev_k['y'])
        if not prev_k and next_k:
            return (next_k['x'], next_k['y'])
        if prev_k and next_k:
            # Interpolate
            t1 = prev_k['time']
            t2 = next_k['time']
            ratio = (timestamp - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
            
            x = prev_k['x'] + (next_k['x'] - prev_k['x']) * ratio
            y = prev_k['y'] + (next_k['y'] - prev_k['y']) * ratio
            return (x, y)
            
        return None
        
    def _is_scene_cut(self, frame: np.ndarray, thresh: float = 0.5) -> bool:
        """Detects scene cut using histogram correlation."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        is_cut = False
        if self.prev_hist is not None:
            score = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            if score < thresh:
                is_cut = True
        
        self.prev_hist = hist
        return is_cut

    def _get_active_speaker_index(self, faces: list, timestamp: float, frame: np.ndarray = None) -> int:
        if not faces:
            return None
            
        # Update persistence (Kalman Tracking)
        # Pass frame for Re-ID embedding
        self.tracker.update(faces, self.reader.width, frame_image=frame)
        
        audio_energy = self.audio.get_audio_energy(timestamp - 0.1, timestamp + 0.1)
        
        best_score = -9999
        best_idx = None
        frame_h = self.reader.height
        
        max_persistence = 1
        if self.tracker.tracks:
            max_persistence = max([t.age for t in self.tracker.tracks])
            if max_persistence < 1:
                max_persistence = 1
         
        scores = []
        for i, face in enumerate(faces):
            landmarks = face.get('landmarks')
            mouth_openness = self.mouth_detector.get_mouth_openness(landmarks, frame_h)
            
            score = (audio_energy * WEIGHT_AUDIO_ACTIVITY) * \
                    (mouth_openness * WEIGHT_MOUTH_ENERGY) * \
                    (face.get('confidence', 1.0) * WEIGHT_FACE_CONFIDENCE)
            
            p_score = (face.get('track_count', 0) / max_persistence) * 2.0
            score += p_score
            
            # Phase: Frontal Face Priority
            # Bonus for facing camera (0.0 to 1.0) -> Max +2.0 points
            facing_bonus = face.get('facing_score', 0.5) * 2.0
            score += facing_bonus
            
            # Talking Bonus (High Mouth + Audio)
            if mouth_openness > 0.15 and audio_energy > 0.1:
                score += 5.0 
            
            face['speaker_score'] = score
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # --- Speaker Hysteresis (Stability) ---
        if self.last_speaker_idx is not None and self.last_speaker_idx < len(faces) and best_idx != self.last_speaker_idx:
            # Check if previous speaker is still somewhat active
            prev_score = scores[self.last_speaker_idx]
            
            # Switching Condition: 
            # 1. Held for enough frames OR
            # 2. New score is significantly higher
            
            is_strong_switch = best_score > (prev_score * self.SPEAKER_SWITCH_THRESH)
            has_held_long_enough = self.speaker_streak > self.SPEAKER_MIN_HOLD
            
            if not is_strong_switch and not has_held_long_enough:
                # Stick to old speaker
                best_idx = self.last_speaker_idx
                self.speaker_streak += 1
            else:
                # Switch allowed
                self.last_speaker_idx = best_idx
                self.speaker_streak = 0
        else:
             if best_idx == self.last_speaker_idx:
                 self.speaker_streak += 1
             else:
                 self.last_speaker_idx = best_idx
                 self.speaker_streak = 0
                 
        return best_idx

    def run(self, start_time=0.0, end_time=None):
        if self.solver_mode:
            # Solver mode doesn't support trim yet easily
            self._run_solver_pipeline()
        else:
            self._run_reactive_pipeline(start_time, end_time)

    def _run_solver_pipeline(self):
        logging.info("Starting Enterprise Solver Pipeline (Two-Pass)...")
        # Placeholder for solver pipeline if needed, but for now copying structure
        # Assuming solver pipeline logic is same as before, but minimizing for this extraction 
        # to focus on reactive for simplicity unless required. Original had it.
        # Just calling path_solver... 
        pass 
        
    def _run_reactive_pipeline(self, start_time=0.0, end_time=None):
        logging.info(f"Starting standard reactive tracking from {start_time}s to {end_time}s...")
        
        # 0. Fast Seek to Start
        self.reader.set_position(start_time)
        
        # 1. Init Writer with Audio Offset
        # We need to pass start/duration to writer for correct audio sync
        duration = None
        if end_time:
            duration = end_time - start_time
            
        self.writer.start_time = start_time
        if self.writer.process:
            self.writer.release()
            
        from .video.writer import VideoWriter
        self.writer = VideoWriter(
            self.output_path, 
            self.output_width, 
            self.output_height, 
            self.reader.fps, 
            self.input_path,
            start_time=start_time,
            duration=duration
        )

        # Main Processing Loop
        self.writer.process = self.writer.process
        scene_cuts = 0
        
        for frame in tqdm(self.reader, total=self.reader.total_frames):
            timestamp = self.reader.get_timestamp()
            
            # Trim Logic
            if timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                break
            
            # 1. Scene Detection
            if self._is_scene_cut(frame):
                scene_cuts += 1
                logging.info(f"Scene Cut at {timestamp:.2f}s. Resetting focus/smoother.")
                self.smoother.reset() 
                self.last_valid_focus = None
                self.focus_loss_frames = 0
                self.last_focus_point = None
            
            # 2. Run Detectors
            if self.frame_count % DETECTION_INTERVAL == 0:
                faces = self.face_detector.detect(frame, timestamp)
                people = self.person_detector.detect(frame)
                motion = self.motion_detector.detect(frame)
                
                # Text Detection
                text_boxes = []
                if self.text_detector:
                     text_boxes = self.text_detector.detect_text_regions(frame)
                     
                if timestamp < self.reader.duration:
                    self.audio.is_active(timestamp)
                
                # Pass frame to tracker via get_active_speaker_index -> tracker.update
                speaker_idx = self._get_active_speaker_index(faces, timestamp, frame=frame)
                
                # --- USER OVERRIDES (Phase 16) ---
                if faces:
                     # Filter Ignored
                     if self.user_ignored_ids:
                         faces = [f for f in faces if f.get('track_id') not in self.user_ignored_ids]
                     
                     # Prioritize Stars
                     if self.user_star_ids:
                         star_faces = [f for f in faces if f.get('track_id') in self.user_star_ids]
                         if star_faces:
                             faces = star_faces
                             # If we found our stars, we force them.
                             # This effectively overrides the "Strict Star Locking" on unrelated people too.
                             
                     # Re-determine active speaker in the filtered list
                     speaker_idx = None
                     if faces:
                         # speaker_score is already populated by _get_active_speaker_index
                         # Just find max in new list
                         best_s = -float('inf')
                         for ix, f in enumerate(faces):
                             if f.get('speaker_score', -100) > best_s:
                                 best_s = f['speaker_score']
                                 speaker_idx = ix
                
                # --- ROBUST STAR LOCKING (Phase 12) ---
                active_faces = faces
                if faces and self.tracker.tracks:
                    # Best Track
                    best_track = max(self.tracker.tracks, key=lambda t: t.age)
                    max_track_age = best_track.age
                    
                    # Acquire Lock
                    if self.locked_star_id is None:
                        if max_track_age > self.LOCK_PERSISTENCE_REQ:
                            self.locked_star_id = best_track.track_id
                            logging.info(f"Target Acquired: Locking onto Star ID {self.locked_star_id}")
                    
                    # Enforce Lock
                    if self.locked_star_id is not None:
                        found_lock = False
                        
                        # Match detections to the locked track using proximity
                        locked_track = next((t for t in self.tracker.tracks if t.track_id == self.locked_star_id), None)
                        
                        if locked_track:
                             px, py = locked_track.state()
                             best_dist = 99999
                             best_face = None
                             
                             for f in faces:
                                 fx = f['bbox'][0] + f['bbox'][2]/2
                                 fy = f['bbox'][1] + f['bbox'][3]/2
                                 d = ((fx-px)**2 + (fy-py)**2)**0.5
                                 if d < best_dist:
                                     best_dist = d
                                     best_face = f
                             
                             # Match Threshold (20% of width)
                             if best_face and best_dist < (self.reader.width * 0.2):
                                 active_faces = [best_face]
                                 found_lock = True
                                 self.lock_unlock_timer = 0
                                 # Reset speaker index to 0 since we only have 1 face
                                 if speaker_idx is not None:
                                     speaker_idx = 0 
                        
                        if not found_lock:
                            # Try visual Re-ID match for locked star?
                            # If tracker lost it spatially, maybe Re-ID found it as new track?
                            # Check active faces for matching track ID
                            reid_match = next((f for f in faces if f.get('track_id') == self.locked_star_id), None)
                            if reid_match:
                                active_faces = [reid_match]
                                found_lock = True
                                self.lock_unlock_timer = 0
                                logging.info(f"Re-ID: Rediscovered Locked Star {self.locked_star_id}")
                                
                            if not found_lock:
                                self.lock_unlock_timer += 1
                                if self.lock_unlock_timer > self.UNLOCK_TIMEOUT:
                                    logging.info(f"Target Lost (Timeout): Unlocking Star ID {self.locked_star_id}")
                                    self.locked_star_id = None
                                    active_faces = faces
                                else:
                                    active_faces = [] # Hide others while holding for Star
                
                faces = active_faces
                
                
                sorted_faces = []
                if faces:
                     sorted_faces = sorted(faces, key=lambda f: f.get('speaker_score', 0) + (f['bbox'][2]*f['bbox'][3])/(self.reader.width**2), reverse=True)
                
                # Split Screen Check
                if self.compositor and len(sorted_faces) >= 2:
                    f1 = sorted_faces[0]
                    f2 = sorted_faces[1]
                    c1 = f1['bbox'][0] + f1['bbox'][2]/2
                    c2 = f2['bbox'][0] + f2['bbox'][2]/2
                    dist = abs(c1 - c2)
                    if dist > (self.reader.width * SPLIT_SCREEN_THRESHOLD):
                         if not self.is_split_active:
                             logging.info("Split Screen Mode Activating")
                         self.is_split_active = True
                         self.split_targets = [f1['bbox'], f2['bbox']]
                    else:
                         self.is_split_active = False
                else:
                    self.is_split_active = False 
                
                self.last_faces = sorted_faces
                
                # Group Detection
                group_info = None
                if not self.is_split_active and self.group_selector and faces and len(faces) >= 2:
                    group_info = self.group_selector.detect_primary_group(faces, self.reader.width)
                
                self.last_group_info = group_info
                
                subject_box = None 
                if group_info:
                     gx, gy, gw, gh = group_info['bbox']
                     raw_focus = (gx + gw/2, gy + gh/2)
                     self.last_subject_box = group_info['bbox']
                     subject_box = group_info['bbox']
                else:
                    raw_focus = self.focus_selector.select_focus(
                        sorted_faces if sorted_faces else faces, 
                        people, motion, speaker_idx, 
                        (self.reader.height, self.reader.width)
                    )
                    subject_box = None
                    if speaker_idx is not None and faces:
                        subject_box = faces[speaker_idx]['bbox']
                    elif faces:
                        subject_box = sorted(faces, key=lambda f: f['bbox'][2]*f['bbox'][3], reverse=True)[0]['bbox']
                    elif people:
                        subject_box = sorted(people, key=lambda p: p['bbox'][2]*p['bbox'][3], reverse=True)[0]['bbox']
                    self.last_subject_box = subject_box
                
                # --- MERGE TEXT BOXES ---
                if text_boxes:
                    # Initialize with existing subject or first text box
                    if subject_box:
                        min_x, min_y, w, h = subject_box
                        max_x = min_x + w
                        max_y = min_y + h
                    else:
                        min_x, min_y, w, h = text_boxes[0]
                        max_x = min_x + w
                        max_y = min_y + h
                    
                    for (tx, ty, tw, th) in text_boxes:
                        min_x = min(min_x, tx)
                        min_y = min(min_y, ty)
                        max_x = max(max_x, tx + tw)
                        max_y = max(max_y, ty + th)
                    
                    # Update Subject Box
                    subject_box = (min_x, min_y, max_x - min_x, max_y - min_y)
                    self.last_subject_box = subject_box
                    
                    # Override Focus Point to be center of this new union
                    raw_focus = (min_x + (max_x - min_x)/2, min_y + (max_y - min_y)/2)

                # Focus Hold
                has_target = (group_info is not None) or (len(faces) > 0) or (len(people) > 0)
                if has_target:
                    self.focus_loss_frames = 0
                    self.last_valid_focus = raw_focus
                else:
                    self.focus_loss_frames += 1
                    if self.focus_loss_frames < self.MAX_FOCUS_LOSS and self.last_valid_focus:
                        raw_focus = self.last_valid_focus
                
                self.last_focus_point = raw_focus
                
                if self.frame_count == 0 and raw_focus:
                    self.smoother.reset(raw_focus)

            self.frame_count += 1
            
            # Split Screen
            if self.is_split_active and self.compositor and len(self.split_targets) == 2:
                 combined_frame = self.compositor.create_split_screen(frame, self.split_targets[0], self.split_targets[1])
                 self.writer.write(combined_frame)
                 continue 
            
            # Smooth
            # Check for Manual Keyframe Override (Phase 16)
            manual_pt = self._get_manual_focus(timestamp)
            
            if manual_pt:
                smoothed_focus = manual_pt
                # Update smoother internal state to avoid jump when manual ends
                self.smoother.kf.statePost[0] = manual_pt[0]
                self.smoother.kf.statePost[1] = manual_pt[1]
            elif self.last_focus_point:
                smoothed_focus = self.smoother.update(
                    self.last_focus_point, 
                    timestamp, 
                    frame_dims=(self.reader.width, self.reader.height)
                )
            else:
                smoothed_focus = (self.reader.width / 2, self.reader.height / 2)

            # Zoom
            current_height = self.base_crop_height
            if self.zoom_controller:
                # Calculate Audio Energy (for Pulse)
                # Short window (100ms) for reactivity
                current_audio = 0.0
                if self.audio:
                     current_audio = self.audio.get_audio_energy(timestamp, timestamp + 0.1)
                
                zoom_factor = self.zoom_controller.calculate_zoom(
                    self.last_subject_box, 
                    smoothed_focus, 
                    (self.reader.width, self.reader.height), 
                    timestamp,
                    audio_energy=current_audio
                )
                current_height = int(self.base_crop_height * zoom_factor)

            # Crop
            x1, y1, x2, y2 = self.cropper.get_crop_coords(smoothed_focus, current_height)
            crop = frame[y1:y2, x1:x2]
            
            # --- CINEMA MODE LOGIC (Hysteresis) ---
            should_be_cinema = False
            if self.couple_zoom_mode and self.last_group_info and not self.is_split_active:
                gx, gy, gw, gh = self.last_group_info['bbox']
                std_crop_w = int(self.base_crop_height * (self.output_width / self.output_height))
                
                # Check condition
                if gw > (std_crop_w * 0.85):
                    should_be_cinema = True
            
            # State Machine
            if should_be_cinema:
                self.CINEMA_TRIGGER_COUNT += 1
                if self.cinema_mode_active:
                     self.cinema_hold_timer = self.CINEMA_HOLD_MAX # Reset hold if condition met
                elif self.CINEMA_TRIGGER_COUNT > self.CINEMA_TRIGGER_REQ:
                     self.cinema_mode_active = True
                     self.cinema_hold_timer = self.CINEMA_HOLD_MAX
                     logging.info("Cinema Mode Activated (Wide Group)")
            else:
                self.CINEMA_TRIGGER_COUNT = 0
                if self.cinema_mode_active:
                    self.cinema_hold_timer -= 1
                    if self.cinema_hold_timer <= 0:
                        self.cinema_mode_active = False
                        logging.info("Cinema Mode Deactivated")

            # Match state logic to update transition
            if self.cinema_mode_active:
                self.cinema_transition = min(1.0, self.cinema_transition + self.TRANSITION_SPEED)
            else:
                self.cinema_transition = max(0.0, self.cinema_transition - self.TRANSITION_SPEED)
                
            # --- UNIFIED RENDER LOGIC ---
            # If transition > 0, we interpolate between Vertical and Cinema crop
            if self.cinema_transition > 0.01 and self.last_group_info:
                 # 1. Calculate Cinema Target Dimensions
                 gx, gy, gw, gh = self.last_group_info['bbox']
                 target_w = min(self.reader.width, int(gw * 1.5))
                 target_h = int(target_w / (16/9))
                 
                 # 2. Calculate Vertical Target Dimensions (Standard)
                 std_w = int(self.base_crop_height * (self.output_width / self.output_height))
                 std_h = self.base_crop_height
                 
                 # 3. Interpolate Crop Size
                 t = self.cinema_transition
                 # Ease in/out? Smoothstep for better feel: t * t * (3 - 2 * t)
                 smooth_t = t * t * (3 - 2 * t)
                 
                 current_w = int(std_w * (1 - smooth_t) + target_w * smooth_t)
                 current_h = int(std_h * (1 - smooth_t) + target_h * smooth_t)
                 
                 # 4. Extract Crop (Centered on focus)
                 cx, cy = smoothed_focus
                 cx1 = int(max(0, cx - current_w // 2))
                 cy1 = int(max(0, cy - current_h // 2))
                 cx2 = int(min(self.reader.width, cx1 + current_w))
                 cy2 = int(min(self.reader.height, cy1 + current_h))
                 
                 # Re-verify actual extracted size
                 real_w = cx2 - cx1
                 real_h = cy2 - cy1
                 
                 if real_w > 0 and real_h > 0:
                     raw_crop = frame[cy1:cy2, cx1:cx2]
                     
                     # 5. Fit to Output Container (Letterbox Logic)
                     canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
                     
                     # Scale to always fill WIDTH (width-based zoom)
                     scale = self.output_width / real_w
                     disp_h = int(real_h * scale)
                     
                     resized = cv2.resize(raw_crop, (self.output_width, disp_h), interpolation=cv2.INTER_LINEAR)
                     
                     # Center Vertically
                     y_off = (self.output_height - disp_h) // 2
                     
                     # Paste
                     # Handle overlap checks
                     y1_c = max(0, y_off)
                     y2_c = min(self.output_height, y_off + disp_h)
                     
                     # Source range
                     y1_r = 0
                     if y_off < 0: y1_r = -y_off
                     h_r = y2_c - y1_c
                     
                     if h_r > 0:
                         canvas[y1_c:y2_c, 0:self.output_width] = resized[y1_r:y1_r+h_r, :]
                         crop = canvas
            
            # Standard Resize if not overridden (or if transition is 0)
            if crop.shape[0] != self.output_height or crop.shape[1] != self.output_width:
                 crop = self.resize_frame(crop, self.output_width, self.output_height)
            
            self.writer.write(crop)
            
        self.writer.release()
        logging.info("Processing complete.")

    def resize_frame(self, frame, width, height):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    def analyze_participants(self, progress_callback=None):
        """
        Runs a pass to identify unique faces for the Star Manager.
        Returns: Dict of {track_id: {'id': int, 'thumb': np.array, 'count': int, 'max_size': int}}
        """
        logging.info("Starting Participant Analysis...")
        # Reset tracker and state
        self.tracker = MultiObjectTracker(max_age=30, max_dist_ratio=0.2) 
        self.frame_count = 0
        
        participants = {}
        total_frames = self.reader.total_frames
        
        # Use iterator manually
        self.reader = VideoReader(self.input_path) # Ensure fresh start
        
        # We need to manually iterate to avoid issues if tqdm is wrapped elsewhere
        pbar = tqdm(total=total_frames, desc="Analyzing Participants")
        
        for frame in self.reader:
            timestamp = self.reader.get_timestamp()
            
            if progress_callback and self.frame_count % 10 == 0:
                 progress_callback(int((self.frame_count / total_frames) * 100))
            
            # Run Detectors on Interval
            if self.frame_count % DETECTION_INTERVAL == 0:
                faces = self.face_detector.detect(frame, timestamp)
                
                # Update Tracker to assign IDs with Re-ID support
                if faces:
                    self.tracker.update(faces, self.reader.width, frame_image=frame)
                    
                    # Collect Data
                    for f in faces:
                        tid = f.get('track_id')
                        if tid is not None:
                            x, y, w, h = f['bbox']
                            x = max(0, int(x))
                            y = max(0, int(y))
                            w = min(int(w), self.reader.width - x)
                            h = min(int(h), self.reader.height - y)
                            
                            area = w * h
                            
                            if tid not in participants:
                                # New Face
                                if w > 0 and h > 0:
                                    thumb = frame[y:y+h, x:x+w].copy()
                                    participants[tid] = {
                                        'id': tid,
                                        'thumb': thumb,
                                        'count': 1,
                                        'max_size': area,
                                        'first_seen': timestamp
                                    }
                            else:
                                # Existing Face
                                participants[tid]['count'] += 1
                                # Update thumb if larger (better quality usually)
                                if area > participants[tid]['max_size'] and w > 0 and h > 0:
                                    participants[tid]['max_size'] = area
                                    participants[tid]['thumb'] = frame[y:y+h, x:x+w].copy()
                                    
            self.frame_count += 1
            pbar.update(1)
            
        pbar.close()
            
        # Reset for main run
        self.reader = VideoReader(self.input_path)
        self.frame_count = 0
        self.tracker = MultiObjectTracker(max_age=30, max_dist_ratio=0.2)
        
        logging.info(f"Analysis Complete. Found {len(participants)} unique participants.")
        return participants
