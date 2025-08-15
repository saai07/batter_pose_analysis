###Cricket Cover Drive Analysis

DESCRIPTION:
This program analyzes a cricket cover drive video to evaluate player biomechanics. 
It uses MediaPipe Pose for human pose estimation and OpenCV for video processing.
The system calculates elbow angle, spine lean, head-to-knee distance, and foot direction, 
provides real-time feedback on the video, and generates a summary report in JSON format.

REQUIREMENTS:
- Python 3.7 or higher
- OpenCV           (pip install opencv-python)
- MediaPipe ==0.10.14  (pip install mediapipe)
- NumPy           (pip install numpy)

FILES:
- cover_drive.py                → Python script containing the program.
- input_video.mp4         → Input video file for analysis (must be in the same directory).
- output/                 → Folder created automatically to store results:
    - annotated_video.mp4 → Video with pose skeleton and metric overlays.
    - evaluation.json     → JSON file with average metrics, scores, and feedback.

USAGE:
1. Install dependencies:
   `pip install -r requirements.txt`

2. Place the cricket batting video in the working directory and name it:
   input_video.mp4

3. Run the program:
   `python cover_drive.py`

4. View results in the `output` folder:
   - `annotated_video.mp4` → Contains pose skeleton and real-time feedback.
   - `evaluation.json`     → Contains metrics, scores, and recommendations.

OUTPUT:
- Annotated video showing:
  * Pose landmarks and skeleton
  * Calculated metrics on screen
  * Feedback messages (elbow angle, spine alignment)
- JSON report with:
  * Average values of metrics
  * Performance scores (1–10 scale)
  * Improvement feedback for each skill

NOTES:
- The analysis assumes the batsman is right-handed.
- The video will be resized if its width is greater than 1280 pixels.
- Metric thresholds can be adjusted in the `add_overlays()` and `evaluate_shot()` functions.


