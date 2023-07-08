# This will be the main controller that triggers all the other features that will be in the modules folder.

import cv2
import mediapipe as mp
import numpy as np
import time
import Modules.HandTrackingModule as HTM
import math
from google.protobuf.json_format import MessageToDict

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
pTime = 0
cap.set(4, hCam)

# Handedness text generation parameters
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

detector = HTM.handDetector(detectionCon=0.7)

while True:

  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  success, img = cap.read()
  allHands, img = detector.findHands(img, draw=True)
  lmList, bbox = detector.findPosition(img, draw=False)
  cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)



  # # prints out the side of hand it sees. Needs to be flipped. 
  # hand_landmarks_list = detector.results.multi_hand_landmarks
  # handedness_list = detector.results.multi_handedness

  # if detector.results.multi_hand_landmarks:
  #   for idx in range(len(hand_landmarks_list)):
  #     hand_landmarks = hand_landmarks_list[idx]
  #     handedness = handedness_list[idx]
  #     for idx, hand_handedness in enumerate(handedness_list):
  #       handedness_dict = MessageToDict(hand_handedness)
  #   cv2.putText(img, f"{handedness_dict['classification'][0]['label']}",
  #               (40, 200), cv2.FONT_HERSHEY_DUPLEX,
  #               FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


  cv2.imshow("Img", img)
  cv2.waitKey(1)
