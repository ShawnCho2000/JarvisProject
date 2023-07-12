# This will be the main controller that triggers all the other features that will be in the modules folder.

import cv2
import mediapipe as mp
import numpy as np
import time
import Modules.HandTrackingModule as HTM
import Modules.HandVolumeModule as HVM
import Modules.HandMouseModule as HMM
import math


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
pTime = 0
cap.set(4, hCam)

detector = HTM.handDetector(detectionCon=0.7)
handVolume = HVM.HandVolume(detector)
handMouse = HMM.HandMouse(detector)

while True:

  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  success, img = cap.read()
  img = detector.findHands(img, draw=True)
  lmList, bbox = detector.findPosition(img, draw=False)

  if (handVolume.volumeSignal(lmList, bbox, handType="Left")):
    img = handVolume.volumeControl(img, lmList, bbox, handType="Right")
  elif(handMouse.mouseSignal(lmList, handType="Right")):
    img = handMouse.mouseControl(img, lmList, handType="Right")



  # Figure out how to create a left hand selector. 
  #   Should find the left hand, then if the left hand makes a gesture. Returns what mode it is currently in
  # Then use that to send lmList to correct module. 

  cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)


  cv2.imshow("Img", img)
  cv2.waitKey(1)
