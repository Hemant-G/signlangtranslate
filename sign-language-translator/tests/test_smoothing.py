import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smoothing import PredictionSmoother

def test_prediction_smoother():
    smoother = PredictionSmoother(window_size=5)
    
    assert smoother.smooth("A", 0.9) == "A"
    assert smoother.smooth("A", 0.9) == "A"
    
    # Introduce a flicker (B with high confidence)
    assert smoother.smooth("B", 0.9) == "A" # A is still majority (2 vs 1)
    
    # Low confidence -> Unknown
    assert smoother.smooth("A", 0.4, threshold=0.7) == "A" # majority still A (A:2, B:1, Unknown:1)

def test_stable_check():
    smoother = PredictionSmoother(window_size=3)
    smoother.smooth("A", 0.9)
    smoother.smooth("A", 0.9)
    
    assert not smoother.is_stable("A", required_frames=3)
    
    smoother.smooth("A", 0.9)
    assert smoother.is_stable("A", required_frames=3)
