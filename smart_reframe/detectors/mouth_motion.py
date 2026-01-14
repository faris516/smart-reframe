class MouthMotionDetector:
    def __init__(self):
        # MediaPipe FaceMesh indices
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14

    def get_mouth_openness(self, landmarks, frame_height: int) -> float:
        """
        Calculates the vertical distance between upper and lower lips.
        Args:
            landmarks: Normalized landmarks from FaceDetector.
            frame_height: Height of the frame in pixels.
        Returns:
            Vertical distance in pixels.
        """
        if not landmarks:
            return 0.0
            
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]
        
        # Calculate vertical distance
        # We use absolute difference in Y (normalized) * height
        dist = abs(upper.y - lower.y) * frame_height
        
        return dist
