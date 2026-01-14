from typing import List, Tuple, Optional
import math
from ..config import MAX_GROUP_DISTANCE_RATIO, MIN_GROUP_CONFIDENCE

class GroupSelector:
    def __init__(self):
        pass

    def detect_primary_group(self, faces: List[dict], frame_width: int) -> Optional[dict]:
        """
        Identifies the most significant group of subjects (e.g. 2 speakers).
        Returns a 'virtual face' dict with a combined bbox if a group is found.
        Returns None if no valid group is found (or single subject is better).
        
        Logic:
        1. Sort faces by importance (speaker_score > size).
        2. Take the top face (Primary).
        3. Look for a "Buddy" (Secondary) face that is:
           - Close enough (distance < threshold).
           - Large enough (relative to Primary).
        4. If found, return Union Bounding Box.
        """
        if len(faces) < 2:
            return None
            
        # Sort by speaker score if available, else by area
        # Assuming 'speaker_score' is populated by pipeline
        sorted_faces = sorted(faces, key=lambda f: f.get('speaker_score', 0) + (f['bbox'][2]*f['bbox'][3])/(frame_width*frame_width), reverse=True)
        
        primary = sorted_faces[0]
        px, py, pw, ph = primary['bbox']
        pcx = px + pw/2
        
        # Max distance in pixels
        # For vertical framing (9:16), we can't fit people who are too far apart.
        # Max zoom 2.0x gives us ~1200px width.
        # So subjects must be within ~900px (0.47) to fit with margins.
        max_dist_px = frame_width * 0.45
        
        if 'speaker_score' in primary:
             # If we have speaker scores, ensure primary has detection confidence
             if primary.get('confidence', 0) < MIN_GROUP_CONFIDENCE:
                 return None
        
        best_buddy = None
        
        # Check other faces
        # We only really care about the 2nd most important face for the 'main duo'
        potential_buddies = sorted_faces[1:]
        
        for other in potential_buddies:
            ox, oy, ow, oh = other['bbox']
            ocx = ox + ow/2
            
            # Distance check
            dx = abs(pcx - ocx)
            
            if dx < max_dist_px:
                # Found a buddy!
                best_buddy = other
                break
        
        if best_buddy:
            # Calculate Union BBox
            ox, oy, ow, oh = best_buddy['bbox']
            
            x1 = min(px, ox)
            y1 = min(py, oy)
            x2 = max(px+pw, ox+ow)
            y2 = max(py+ph, oy+oh)
            
            union_w = x2 - x1
            union_h = y2 - y1
            
            return {
                'bbox': (x1, y1, union_w, union_h),
                'confidence': (primary.get('confidence', 1.0) + best_buddy.get('confidence', 1.0)) / 2,
                'is_group': True,
                'members': [primary, best_buddy]
            }
            
        return None
