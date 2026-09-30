import cv2
import time
import sys
import os
import joblib
import numpy as np

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_detector import HandDetector
from src.features import extract_features
from src.smoothing import PredictionSmoother
from src.text_buffer import TextBuffer
from src.tts import TTS

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=2)
    
    smoother = PredictionSmoother(window_size=7)
    text_buffer = TextBuffer(debounce_frames=10)
    tts = TTS()
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model_path = os.path.join(models_dir, "sign_classifier.joblib")
    
    classifier = None
    if os.path.exists(model_path):
        classifier = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No trained model found in models/ directory.")
        print("Run `python training/collect_data.py` and `python training/train.py` first.")
    
    p_time = 0
    print("Starting webcam... Press S(speak), C(clear), B(backspace), SPACE, Q/ESC(quit).")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1) # Mirror image
        
        landmarks_list = detector.process(img, draw=True)
        
        current_prediction = "Unknown"
        confidence = 0.0
        
        if landmarks_list and classifier:
            features = extract_features(landmarks_list[0])
            probas = classifier.predict_proba([features])[0]
            max_idx = np.argmax(probas)
            current_prediction = classifier.classes_[max_idx]
            confidence = probas[max_idx]
            
        smoothed_pred = smoother.smooth(current_prediction, confidence, threshold=0.7)
        
        if smoother.is_stable(smoothed_pred, required_frames=5) and smoothed_pred != "Unknown":
            text_buffer.append(smoothed_pred)
        elif smoothed_pred == "Unknown":
            text_buffer.append("Unknown")
            
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        cv2.rectangle(img, (0, 0), (img.shape[1], 150), (30, 30, 30), cv2.FILLED)
        cv2.putText(img, f"Prediction: {smoothed_pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {confidence*100:.1f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.rectangle(img, (0, img.shape[0]-80), (img.shape[1], img.shape[0]), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f"Text: {text_buffer.get_text()}", (20, img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.imshow("Real-Time Sign Language Translator", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            text_buffer.clear()
        elif key == ord('b'):
            text_buffer.backspace()
        elif key == 32: # SPACE
            text_buffer.space()
        elif key == ord('s'):
            text_to_speak = text_buffer.get_text()
            print(f"Speaking: {text_to_speak}")
            tts.speak(text_to_speak)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
