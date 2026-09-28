from collections import deque
from collections import Counter

class PredictionSmoother:
    def __init__(self, window_size=7):
        """
        Maintains a rolling window of predictions and returns the majority vote.
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def smooth(self, prediction, confidence, threshold=0.70):
        # Ignore low confidence
        if confidence < threshold:
            prediction = "Unknown"
            
        self.history.append(prediction)
        
        # If history isn't full yet or is empty
        if not self.history:
            return "Unknown"
            
        counter = Counter(self.history)
        majority_pred, count = counter.most_common(1)[0]
        
        return majority_pred
        
    def is_stable(self, prediction, required_frames=5):
        """
        Check if the last `required_frames` frames match the prediction exactly.
        """
        if len(self.history) < required_frames:
            return False
            
        # Extract last N frames
        recent_frames = list(self.history)[-required_frames:]
        return all(p == prediction for p in recent_frames)
