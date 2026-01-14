import pytest
import numpy as np
from smart_reframe.tracking.tracker import KalmanTrack

def test_static_ghost_detection():
    # 1. Initialize Track
    bbox = (100, 100, 50, 50)
    track = KalmanTrack(track_id=1, initial_bbox=bbox)
    
    # 2. Simulate 40 frames of STATIC mouth (0.0) -> Should eventually be ghost
    # Requirement is age >= 30 and low history
    for _ in range(40):
        track.predict()
        track.update(bbox, mouth_val=0.0)
        
    assert track.age == 40
    assert len(track.mouth_history) == 40
    
    # Should be True (Avg=0, Max=0)
    assert track.is_static_ghost() is True
    print("Static face correctly flagged as Ghost.")

def test_dynamic_person_detection():
    # 1. Initialize Track
    bbox = (100, 100, 50, 50)
    track = KalmanTrack(track_id=2, initial_bbox=bbox)
    
    # 2. Simulate 40 frames of DYNAMIC mouth (0.0 to 0.2)
    for i in range(40):
        track.predict()
        # Varying mouth
        val = 0.0
        if i % 10 == 0: val = 0.2
        track.update(bbox, mouth_val=val)
        
    # Should be False (Max=0.2 > 0.05)
    assert track.is_static_ghost() is False
    print("Dynamic face correctly identified as Real.")

def test_short_duration_safety():
    # 1. Initialize Track
    bbox = (100, 100, 50, 50)
    track = KalmanTrack(track_id=3, initial_bbox=bbox)
    
    # 2. Simulate only 5 frames of 0.0
    for _ in range(5):
        track.predict()
        track.update(bbox, mouth_val=0.0)
        
    # Should be False (Age < 30)
    # We don't want to instant-ban a face just because it's silent for 0.1s
    assert track.is_static_ghost() is False
    print("Short duration safely ignored.")

if __name__ == "__main__":
    test_static_ghost_detection()
    test_dynamic_person_detection()
    test_short_duration_safety()
