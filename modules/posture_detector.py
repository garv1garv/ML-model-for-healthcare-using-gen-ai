
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from typing import Tuple, Optional, List

class AdvancedPostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=2
        )
        
        
        self.posture_threshold = 80   
        self.neck_angle_threshold = 140
        self.shoulder_level_threshold = 0.05  
        self.hip_level_threshold = 0.05
        self.smoothing_window = 15
        self.posture_history = deque(maxlen=self.smoothing_window)
        
       
        self.frame_count = 0
        self.start_time = time.time()
        
      
        self.feedback_colors = {
            'good': (0, 255, 0),
            'poor': (0, 0, 255),
            'warning': (0, 255, 255)
        }
    
        self.poor_posture_duration = 0
        self.last_poor_posture_time = None
        self.posture_warning_active = False

    def _calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate the angle between three points in degrees."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_landmark_position(self, landmarks, landmark_id: int, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert normalized landmark to pixel coordinates."""
        landmark = landmarks[landmark_id]
        if landmark.visibility < 0.6:
            return None
        return (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Analyze posture in a single frame."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        posture_status = "Good Posture"
        feedback_color = self.feedback_colors['good']
        issues: List[str] = []
        frame_height = frame.shape[0]

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            # Get relevant landmarks
            left_shoulder = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, (h, w))
            right_shoulder = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, (h, w))
            left_hip = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, (h, w))
            right_hip = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, (h, w))
            left_knee = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, (h, w))
            left_ear = self._get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR, (h, w))

            # Hip angle analysis (left side)
            if all([left_shoulder, left_hip, left_knee]):
                hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
                self.posture_history.append(hip_angle)
                
                if len(self.posture_history) == self.smoothing_window:
                    smoothed_angle = np.mean(self.posture_history)
                    if smoothed_angle < self.posture_threshold:
                        issues.append(f"Hip angle: {smoothed_angle:.1f}°")

            # Neck angle analysis (left side)
            if all([left_ear, left_shoulder, left_hip]):
                neck_angle = self._calculate_angle(left_ear, left_shoulder, left_hip)
                if neck_angle < self.neck_angle_threshold:
                    issues.append(f"Neck angle: {neck_angle:.1f}°")

            # Shoulder alignment check
            if left_shoulder and right_shoulder:
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / h
                if shoulder_diff > self.shoulder_level_threshold:
                    issues.append(f"Shoulders uneven: {shoulder_diff*100:.1f}%")

            # Hip alignment check
            if left_hip and right_hip:
                hip_diff = abs(left_hip[1] - right_hip[1]) / h
                if hip_diff > self.hip_level_threshold:
                    issues.append(f"Hips uneven: {hip_diff*100:.1f}%")

            # Posture status determination
            if issues:
                posture_status = "Poor Posture: " + ", ".join(issues)
                feedback_color = self.feedback_colors['poor']
                current_time = time.time()
                
                if self.last_poor_posture_time is None:
                    self.last_poor_posture_time = current_time
                else:
                    self.poor_posture_duration = current_time - self.last_poor_posture_time
                    
                    if self.poor_posture_duration > 10:
                        self.posture_warning_active = True
                        feedback_color = self.feedback_colors['warning']
                        posture_status = "Warning: " + posture_status
            else:
                self.last_poor_posture_time = None
                self.poor_posture_duration = 0
                self.posture_warning_active = False

        return frame, posture_status, feedback_color

    def get_performance_metrics(self) -> dict:
        """Get performance metrics."""
        elapsed_time = time.time() - self.start_time
        return {
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time
        }

def run_posture_detection():
    """Run real-time posture detection."""
    analyzer = AdvancedPostureAnalyzer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting posture detection. Press 'Q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        processed_frame, status, color = analyzer.analyze_frame(frame)
        analyzer.frame_count += 1
        
        # Display posture feedback
        cv2.putText(processed_frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display performance metrics
        metrics = analyzer.get_performance_metrics()
        fps_text = f"FPS: {metrics['fps']:.1f}"
        cv2.putText(processed_frame, fps_text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Advanced Posture Detection', processed_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Posture detection stopped.")

__all__ = ['run_posture_detection']