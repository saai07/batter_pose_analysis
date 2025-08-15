import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from collections import defaultdict

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path, output_path='output/annotated_video.mp4'):
    """Process video frame-by-frame for cover drive analysis"""
    
    os.makedirs('output', exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Normalize resolution for better performance
    if width > 1280:
        scale = 1280 / width
        width, height = 1280, int(height * scale)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize metrics storage with defaultdict for automatic key creation
    metrics_history = defaultdict(list)
    
    start_time = time.time()
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1 
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame if needed
            if (width, height) != frame.shape[:2][::-1]:
                frame = cv2.resize(frame, (width, height))
                
            # Convert to RGB and process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Draw pose skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0)),  # Joint color
                    mp_drawing.DrawingSpec(color=(255, 0, 0))  # Connection color
                )

                # Extract and process keypoints
                keypoints = get_keypoints(results.pose_landmarks.landmark)
                metrics = compute_metrics(keypoints)
                update_metrics_history(metrics, metrics_history)
                add_overlays(frame, metrics)

            
            out.write(frame)
            frame_count += 1

    
    cap.release()
    out.release()

    # Performance metrics
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processing complete. Average FPS: {avg_fps:.2f}")

    # Generate final evaluation
    evaluation = evaluate_shot(metrics_history)
    with open('output/evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=4)

    return output_path

def get_keypoints(landmarks):
    """Extract and normalize keypoint coordinates"""
    return {
        'nose': (landmarks[mp_pose.PoseLandmark.NOSE.value].x, 
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y),
        'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
        'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
        'left_elbow': (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
        'right_elbow': (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
        'left_wrist': (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
        'right_wrist': (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
        'left_hip': (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
        'right_hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
        'left_knee': (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
        'right_knee': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
        'left_ankle': (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
        'right_ankle': (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
    }

def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def compute_metrics(kp):
    """Calculate biomechanical metrics for cover drive"""
    # Right side metrics (assuming right-handed batsman)
    elbow_angle = calculate_angle(
        kp['right_shoulder'], 
        kp['right_elbow'], 
        kp['right_wrist']
    )
    
    # Spine lean relative to vertical
    spine_lean = calculate_angle(
        (kp['right_hip'][0], 0),  # Point directly below hip
        kp['right_hip'],
        kp['right_shoulder']
    )
    
    # Head-knee alignment (horizontal distance)
    head_knee_dist = abs(kp['nose'][0] - kp['right_knee'][0])
    
    # Foot direction (relative to horizontal)
    foot_dir = calculate_angle(
        (kp['right_ankle'][0] - 0.1, kp['right_ankle'][1]),  # Small offset for horizontal reference
        kp['right_ankle'],
        kp['right_knee']
    )
    
    return {
        'elbow_angle': elbow_angle,
        'spine_lean': spine_lean,
        'head_knee_dist': head_knee_dist,
        'foot_dir': foot_dir
    }

def update_metrics_history(metrics, history):
    """Safely update metrics history with new measurements"""
    for key, val in metrics.items():
        if val is not None:
            history[key].append(val)

def add_overlays(frame, metrics):
    """Add visual overlays to the frame"""
    y_pos = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Display metrics
    for key, val in metrics.items():
        if val is not None:
            text = f"{key.replace('_', ' ').title()}: {val:.1f} degree" if 'angle' in key or 'dir' in key else f"{key.replace('_', ' ').title()}: {val:.3f}"
            cv2.putText(frame, text, (10, y_pos), font, 0.5, (0, 255, 0), 2)
            y_pos += 30

    # Add feedback based on thresholds
    if metrics.get('elbow_angle'):
        if 100 <= metrics['elbow_angle'] <= 140:
            cv2.putText(frame, "Good elbow elevation", (10, y_pos), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Adjust elbow angle (100-140 degree)", (10, y_pos), font, 0.5, (0, 0, 255), 2)
        y_pos += 30
        
    if metrics.get('spine_lean'):
        if metrics['spine_lean'] < 15:
            cv2.putText(frame, "Good spine alignment", (10, y_pos), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Reduce spine lean (<15 degree)", (10, y_pos), font, 0.5, (0, 0, 255), 2)
    


 
def evaluate_shot(history):
    """Generate final evaluation report"""
    # Calculate averages with fallback for empty lists
    avg_metrics = {
        'elbow_angle': np.mean(history['elbow_angle']) if history['elbow_angle'] else 0,
        'spine_lean': np.mean(history['spine_lean']) if history['spine_lean'] else 0,
        'head_knee_dist': np.mean(history['head_knee_dist']) if history['head_knee_dist'] else 0,
        'foot_dir': np.mean(history['foot_dir']) if history['foot_dir'] else 0
    }
    
    # Calculate scores (1-10)
    scores = {
        'Footwork': min(10, max(1, 10 - abs(avg_metrics['foot_dir'] - 85)/5)),  # 85° ideal
        'Head Position': 10 if avg_metrics['head_knee_dist'] < 0.1 else 5,
        'Swing Control': min(10, max(1, (avg_metrics['elbow_angle'] - 100)/4)),  # 100-140° range
        'Balance': 10 - min(9, avg_metrics['spine_lean']/5),  # Penalize >5° lean
        'Follow-through': 8  # Placeholder for actual follow-through metrics
    }
    
    # Generate feedback
    feedback = {
        'Footwork': "Stable base" if scores['Footwork'] > 7 else "Improve foot positioning",
        'Head Position': "Excellent head position" if scores['Head Position'] > 7 else "Keep head over knee",
        'Swing Control': "Good swing path" if scores['Swing Control'] > 7 else "Work on elbow position",
        'Balance': "Great balance" if scores['Balance'] > 7 else "Maintain better balance",
        'Follow-through': "Complete follow-through" if scores['Follow-through'] > 7 else "Extend follow-through"
    }
    
    return {
        'average_metrics': avg_metrics,
        'scores': scores,
        'feedback': feedback,
        'frame_count': len(history['elbow_angle']) if history['elbow_angle'] else 0
    }

if __name__ == '__main__':
    input_video = "input_video.mp4" 
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    output_path = process_video(input_video)
    print(f"Analysis complete. Output saved to: {output_path}")