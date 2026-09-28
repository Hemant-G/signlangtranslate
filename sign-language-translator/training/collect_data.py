import cv2
import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_detector import HandDetector
from src.features import extract_features

def main():
    label = input("Enter gesture label: ").strip().upper()
    if not label:
        print("Label cannot be empty.")
        return

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "landmarks.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label'] + [f'feature_{i}' for i in range(1, 64)]
            writer.writerow(header)

        cap = cv2.VideoCapture(0)
        detector = HandDetector(max_num_hands=1)
        
        print(f"Collecting data for label: {label}")
        print("Press SPACE to capture.")
        print("Press Q to quit.")
        
        sample_count = 0
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)
            landmarks_list = detector.process(img, draw=True)
            
            # Display Instructions
            cv2.putText(img, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, f"Samples: {sample_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, "Press SPACE to capture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Press Q to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Data Collection", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # SPACE
                if landmarks_list:
                    features = extract_features(landmarks_list[0])
                    row = [label] + features.tolist()
                    writer.writerow(row)
                    sample_count += 1
                    f.flush()
                    
                    # Visual feedback mapping
                    cv2.putText(img, "CAPTURED!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow("Data Collection", img)
                    cv2.waitKey(200) # Small delay to show feedback
                else:
                    print("No hand detected. Cannot capture.")
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
