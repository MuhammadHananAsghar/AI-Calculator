#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install pyshine')


# In[2]:


import cv2
import math
import numpy as np
import pyshine as ps
import mediapipe as mp


# In[3]:


# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


# In[4]:


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 0.6
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2


# In[5]:


INDEX_A = 8
INDEX_B = 6
MIDDLE_A = 12
MIDDLE_B = 10
RING_A = 16
RING_B = 14
PINKY_A = 20
PINKY_B = 18
THUMB_A = 4
THUMB_B = 2

def cubes_dist(text):
    id = 6
    encodings = "0 1 2 3 4 5 6 7 8 9 * + - / = CLEAR".split()
    x, y, cubes = 25, 30, []
    for index in range(len(encodings)):
        if (index%id == 0) and (index != 0):
            y += 90
            x = 25
        if text == encodings[index]:
            cubes.append([x, y, encodings[index], [228, 31, 40]])
        else:
            cubes.append([x, y, encodings[index], [9, 210, 51]])

        x += 100
    return cubes

def min_dist(points):
    id = 6
    encodings = "0 1 2 3 4 5 6 7 8 9 * + - / = CLEAR".split()
    x, y, cubes, dists = 25, 30, [], []
    for index in range(len(encodings)):
        if (index%id == 0) and (index != 0):
            y += 90
            x = 25
        cubes.append([x, y])
        x += 100
    
    for cube in cubes:
        euc = math.dist(points, cube)
        dists.append(euc)
    return encodings[np.argmin(dists)]


def checkFingers(points):
    fingers = []
    condition = None
    if points[INDEX_A][1] < points[INDEX_B][1]:
        fingers.append(1)
    if points[MIDDLE_A][1] < points[MIDDLE_B][1]:
        fingers.append(1)
    if points[RING_A][1] < points[RING_B][1]:
        fingers.append(1)
    if points[PINKY_A][1] < points[PINKY_B][1]:
        fingers.append(1)
    if points[THUMB_A][0] < points[THUMB_B][0]:
        fingers.append(1)
    if sum(fingers) == 0:
        condition = "ANS"
    if sum(fingers) == 5:
        condition = "CLEAR"
    return condition

def solution(equa):
    for char in equa:
        if char in ["+", "-", "/", "*"]:
            sign = char
    nums = equa.split(sign)
    if sign == "+":
        value = int(nums[0]) + int(nums[1])
    if sign == "-":
        value = int(nums[0]) - int(nums[1])
    if sign == "*":
        value = int(nums[0]) * int(nums[1])
    if sign == "/":
        value = int(nums[0]) / int(nums[1])
    return value


# In[12]:


stream = cv2.VideoCapture(0)
window_name = "AI Calculator"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL);
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

frameCX = 0
overlap = False
size = 20
size = 20
text_value = None
sent = ""
answer = ""

while True:
    ret, frame = stream.read()
    image = cv2.flip(frame, 1)
    img_h, img_w = image.shape[:2]
    if not ret:
        break
    
        # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks:
        
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            image = cv2.blur(image, (20, 20))
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
            points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype("int") for p in hand_landmarks.landmark])
        
        image = cv2.circle(image, points[8], radius=10, color=(0, 0, 255), thickness=-1)
    
        if (points[PINKY_A][1] < points[PINKY_B][1]) and (frameCX == 0):
            text_value = min_dist(points[8])
            if text_value == "CLEAR":
                sent = sent.rstrip(sent[-1])
                sent = sent
            else:
                sent += text_value

            frameCX = 1

        if frameCX != 0:
            frameCX += 1
            if frameCX > 7:
                frameCX = 0
        
        try:
            cond = checkFingers(points)  
            if cond == "ANS":
                eq = sent
                equa = eq.rstrip("=")
                answer = str(solution(equa))
            if cond == "CLEAR":
                sent = ""
                answer = ""
        except:
            pass
    
    distances = cubes_dist(text_value)
    for index, distance in enumerate(distances):
        bgColor = tuple(distance[3])
        image = ps.putBText(image, distance[2], text_offset_x=distance[0],text_offset_y=distance[1],vspace=size,hspace=size, font_scale=2.0,background_RGB=bgColor,text_RGB=(255,255,255))
    
    image = ps.putBText(image, sent+answer, text_offset_x=310,text_offset_y=400,vspace=50,hspace=200, font_scale=1.0,background_RGB=(9, 210, 51),text_RGB=(255,255,255))
    

    
    cv2.imshow(window_name, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
stream.release()
cv2.destroyAllWindows()


# In[16]:


stream.release()
cv2.destroyAllWindows()


# In[ ]:




