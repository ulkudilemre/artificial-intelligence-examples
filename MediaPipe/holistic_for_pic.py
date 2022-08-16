# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import math
import mediapipe as mp

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(name, image):
    h, w = image.shape[:2]
    if h < w:
      img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    
    
path = r"C:\Users\Dilemre\Desktop\Yeni klasör"
os.chdir(path)

names = []
for (dirpath, dirnames, filenames) in os.walk(path):
    names.extend(filenames)
    break

images = {name : cv2.imread(r''+name) for name in names}

# for name, image in images.items():
#     resize_and_show(name, image)
    
    
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=1) as holistic:
    for name, image in images.items():
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          
        # Print nose coordinates.
        image_hight, image_width, _ = image.shape
        if results.pose_landmarks:
            print(
              f'Nose coordinates: ('
              f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
              f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
            )
          
        # Draw pose landmarks.
        print(f'Pose landmarks of {name}:')
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        resize_and_show(name, annotated_image)
        
        
        print('Nose world landmark:'),
        print(results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])
        print(f'Pose world landmarks of {name}:')
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

        
          
         
