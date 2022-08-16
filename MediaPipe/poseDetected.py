# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:26:58 2022

@author: Dilemre
"""

import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

def detectPose(image, pose, display=True):
    """
    This function oerfoms pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value thet is if set to true the function displays the original input
                 image, the resultant and the pose landmarks in 30 plot and returns nating.
                 
    Returns:
        output_image: The input image with the detected pose landmakrs drawn.
        landmarks: A list of detected landmaks converted into their original scale.
    """
    
    output_image = image.copy()
    imageRBG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRBG)
    height, width, _ = image.shape
    
    landmarks = []
    if results.pose_landmarks:
        
        mp_drawing.draw_landmarks(image=output_image, 
                                  landmark_list=results.pose_landmarks, 
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), 
                              int(landmark.y * height), 
                              int(landmark.z * width) ))
            
    if display:
        
        plt.figure(figsize=[22,22])
        plt.subplot(121)
        plt.imshow(image[:,:,::-1])
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis("off")
        
        mp_drawing.plot_landmarks(results.pose_world_landmarks,
                          mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks
  
    
mp_pose = mp.solutions.pose
    
pose_video = mp_pose.Pose(static_image_mode=False, 
                     min_detection_confidence=0.5, 
                     model_complexity=1)

mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)
cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

video.set(3, 1280)
video.set(4, 960)

time1=0

while video.isOpened():
    ok, frame = video.read() 
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame, _ = detectPose(frame, pose_video, display=False)
    time2 = time()
    
    if (time2 - time1) > 0:
        frames_per_sec = 1.0 / (time2 - time1)
        cv2.putText(frame, "FPS {}".format(int(frames_per_sec)),
               (10, 30), cv2.FONT_HERSHEY_PLAIN,
               2, (0, 255, 0), 3)
    time1 = time2
    cv2.imshow("Pose Detection", frame)
    k = cv2.waitKey(1) & 0xFF
    if(k == 27):
        break
video.release()

cv2.destroyAllWindows()
    