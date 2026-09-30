import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        
        hands_landmarks = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_lms.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))
                hands_landmarks.append(landmarks)
                
        return hands_landmarks
