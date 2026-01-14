from ultralytics import YOLO
import logging
import numpy as np

class PersonDetector:
    def __init__(self, model_size='n'):
        """
        Initializes YOLOv8 person detector.
        args:
            model_size: 'n' for nano, 's' for small, etc.
        """
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects people in the frame.
        Returns list of dicts: {'bbox': (x, y, w, h), 'confidence': float}
        """
        if self.model is None:
            return []
            
        # Run inference, class 0 is person
        results = self.model(frame, classes=[0], verbose=False)
        
        people = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates in (x1, y1, x2, y2) format
                b = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = b
                
                people.append({
                    'bbox': (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    'confidence': float(box.conf)
                })
        return people
