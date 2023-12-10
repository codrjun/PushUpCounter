import cv2
from picamera2 import Picamera2
import time
import numpy as np
import mediapipe as mp

picam2=Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 60
picam2.preview_configuration.main.align()
picam2.configure("preview")
picam2.start()
fps = 0

mp_drawing = mp.solutions.drawing_utils
drawSpecific = mp.solutions.pose
mp_pose = mp.solutions.pose

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1, y1) and (x2, y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

pushUpStart = 0
pushUpCount = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while True:
        tStart = time.time()
        im = picam2.capture_array()

        image_height, image_width, _ = im.shape
        results = pose.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            nosePoint = (int(results.pose_landmarks.landmark[0].x * image_width), int(results.pose_landmarks.landmark[0].y * image_height))
            leftWrist = (int(results.pose_landmarks.landmark[15].x * image_width), int(results.pose_landmarks.landmark[15].y * image_height))
            rightWrist = (int(results.pose_landmarks.landmark[16].x * image_width), int(results.pose_landmarks.landmark[16].y * image_height))
            leftShoulder = (int(results.pose_landmarks.landmark[11].x * image_width), int(results.pose_landmarks.landmark[11].y * image_height))
            rightShoulder = (int(results.pose_landmarks.landmark[12].x * image_width), int(results.pose_landmarks.landmark[12].y * image_height))

            if distanceCalculate(rightShoulder, rightWrist) > 370:
                pushUpStart = 1
            elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) < 250:
                pushUpCount = pushUpCount + 1
                pushUpStart = 0

            cv2.putText(im, "Push-up count: " + str(pushUpCount), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
            
            
            #cv2.putText(im, "RW: " + str(distanceCalculate(rightShoulder, rightWrist)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
        
            mp_drawing.draw_landmarks(
                im,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        
        cv2.putText(im, str(int(fps)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Camera", im)

        if cv2.waitKey(1) == ord('q'):
            break

        tEnd = time.time()
        loopTime = tEnd - tStart
        fps = 0.9 * fps + 0.1 * (1 / loopTime)

cv2.destroyAllWindows()
