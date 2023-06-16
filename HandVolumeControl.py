import cv2
import mediapipe as mp
import numpy as np
import time
import Modules.HandTrackingModule as HTM
import math

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
pTime = 0
cap.set(4, hCam)

detector = HTM.handDetector(detectionCon=0.7)

while True:


  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  success, img = cap.read()
  img = detector.findHands(img)
  lmList, bbox = detector.findPosition(img, "right", draw=False)
  if len(lmList) > 0:
    # print(lmList[4], lmList[8])

    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
    cv2.circle(img, (x1, y1), 15, (255,0,0), -1)
    cv2.circle(img, (x2, y2), 15, (255,0,0), -1)
    cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 15, (255,0,0), -1)
    cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)

    length = math.hypot(x2-x1, y2-y1)
    # print(length)
    vol = length - 20
    detector.setVolume(length - 20)
    if vol > 100:
      vol = 100
    if vol < 0:
      vol = 0
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, 400 - int(2.5 * int(vol))), (85, 400), (255, 0, 0), -1)
    cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

  cv2.imshow("Img", img)
  cv2.waitKey(1)
