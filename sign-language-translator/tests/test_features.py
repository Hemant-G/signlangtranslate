import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features

def test_extract_features_shape():
    # 21 random landmarks
    landmarks = np.random.rand(21, 3)
    features = extract_features(landmarks)
    assert features.shape == (63,)

def test_extract_features_wrist_origin():
    landmarks = np.random.rand(21, 3)
    
    features = extract_features(landmarks)
    
    # Since wrist is first, wrist should be at 0, 0, 0
    np.testing.assert_allclose(features[0:3], [0.0, 0.0, 0.0], atol=1e-7)

def test_extract_features_scale_invariant():
    # Base landmarks
    landmarks = np.random.rand(21, 3)
    features1 = extract_features(landmarks)
    
    # Scaled landmarks by factor and translated
    scaled_landmarks = (landmarks * 5.0) + np.array([20.0, -10.0, 5.0])
    features2 = extract_features(scaled_landmarks)
    
    np.testing.assert_allclose(features1, features2, atol=1e-7)
