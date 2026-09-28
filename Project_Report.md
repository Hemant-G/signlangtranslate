# PROJECT PROGRESS REPORT
## Final Year Minor Project — B.E. Computer Science & Engineering

**Student Name / Roll No.:**
- Bhupesh (UE233027)
- Harsh Singh (UE233044)
- Hemant Singh (UE233045)

**Program / Branch:** B.E. (Computer Science & Engineering)  
**Semester:** 7th  
**Guide:** Dr. Deepti Gupta  
**Department:** Computer Science & Engineering

---

## 1. Title
**Real-Time Sign Language Translator:** A computer vision application using OpenCV and MediaPipe to detect hand gestures via webcam and translate them into text and speech in real time.

## 2. Brief Introduction
**Background**  
Sign language is the primary mode of communication for millions of people who are deaf or hard of hearing. Advances in computer vision and lightweight machine learning have made it possible to bridge this communication gap using standard consumer hardware. This project leverages MediaPipe for real-time hand-landmark detection and OpenCV for video processing, coupled with a lightweight classification model to turn a webcam into a gesture-recognition sensor.

**Scope and Assumptions**  
The project targets recognition of a defined static gesture set (alphabets and common words) using a single webcam under controlled lighting. Continuous fluid sign language and complex grammatical structures are out of scope.

## 3. Aim and Objectives
**Aim**  
To develop a real-time computer vision application that detects hand gestures via webcam, classifies them, and translates them into readable text and audible speech.

**Completed Objectives**  
- [x] Implemented real-time hand detection and 21-point landmark extraction (`hand_detector.py`).
- [x] Designed a normalized feature representation (`features.py`).
- [x] Developed data collection and classification model training (`collect_data.py`, `train.py`).
- [x] Implemented temporal smoothing and sequencing to assemble predictions into coherent strings (`smoothing.py`, `text_buffer.py`).
- [x] Integrated a text-to-speech module (`tts.py`).
- [x] Built the master real-time user interface via OpenCV (`app.py`).

## 4. Work Completed & Implementation Details

Following the planned Gantt Chart and Modules, the group has successfully implemented all core milestones leading up to an end-to-end Minimal Viable Product (MVP). The active source codebase is organized as follows:

- **Module 1 & 7 (Video Capture & User Interface):** 
  Implemented in `src/app.py`. Integrates all pieces, rendering bounding boxes, landmark points, and recognized texts on the live OpenCV frame.
  
- **Module 2 (Hand Landmark Detection):** 
  Implemented in `src/hand_detector.py`. Successfully uses MediaPipe Hands for multi-hand landmark extraction.

- **Module 3 (Feature Extraction):** 
  Implemented in `src/features.py`. Flattens and normalizes the raw coordinates to maintain scale and translation invariance.

- **Module 4 (Data Collection & Gesture Classification):**
  - Dataset generation automated via `training/collect_data.py`.
  - Machine learning classifier training script provided via `training/train.py`, utilizing Scikit-learn (Random Forest) for robust gesture mapping.

- **Module 5 (Temporal Smoothing & Sequence Buffer):**
  Implemented in `src/smoothing.py` and `src/text_buffer.py`. Implements debounce filtering and logic for word completion based on idle frames.

- **Module 6 (Text-to-Speech):**
  Implemented via `src/tts.py` utilizing asynchronous TTS playback, preventing video frame blockages.

## 5. Summary & Next Steps
**Current Status:** All functional deliverables outlined in the synopsis have been implemented in code. The pipeline correctly handles live video ingestion, tracking, inference, string accumulation, and auditory output. 

**Next Steps & Optimization Phase (Weeks 13 - 18):**
- Further refinement of test datasets and classification parameters to improve bounding accuracy under diverse lighting.
- Extensive end-user testing to measure pipeline latency and model robustness against complex gesture transitions.
- Adding comprehensive documentation, finalizing test suites, and preparing the final defense demo.
