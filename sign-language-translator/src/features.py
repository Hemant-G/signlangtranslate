import numpy as np

def extract_features(landmarks):
    """
    Extracts normalized features from a list of (x, y, z) MediaPipe landmarks.
    
    1. Subtracts wrist (landmarks[0]) from all points to translate to origin.
    2. Normalizes for scale based on the maximum distance to the wrist.
    
    Args:
        landmarks (list or np.array): 21 list of tuples (x, y, z)
        
    Returns:
        np.array: 63-dimensional flattened feature vector.
    """
    if len(landmarks) != 21:
        raise ValueError("Expected 21 landmarks.")
        
    landmarks = np.array(landmarks)
    
    # Translation: make wrist the origin
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    
    # Scale normalization: max distance from wrist
    max_val = np.max(np.abs(normalized_landmarks))
    if max_val > 0:
        normalized_landmarks = normalized_landmarks / max_val
        
    # Flatten to 63-dimensional vector
    return normalized_landmarks.flatten()
