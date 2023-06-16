import cv2
import mediapipe as mp
import numpy as np
import Modules.HandTrackingModule as htm
import time
from screeninfo import get_monitors
from pynput.mouse import Button, Controller

cap = cv2.VideoCapture(0)

############################################
wCam, hCam = 640, 480
wScr, hScr = get_monitors()[0].width, get_monitors()[0].height
mouse = Controller()
frameR = 100
mouseSmooth = 5
# mouse.position = (wScr/2, hScr/2)
############################################
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(maxHands = 2, detectionCon = 0.9)
prevX, prevY = 0, 0
curX, curY = 0, 0

while True:
  success, img = cap.read()
  img = detector.findHands(img)
  lmList, bbox = detector.findPosition(img, "right")
  cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam - frameR), (255, 0, 255), 2)

  if len(lmList) != 0:
    x1, y1 = lmList[8][1:]
    x2, y2 = lmList[12][1:]
    fingers = detector.fingersUp()
    # print(fingers)

    if fingers[2]==1 and fingers[0] == 0:
      x3, y3 = np.interp(x2, (frameR, wCam - frameR), (0, wScr)), np.interp(y2, (frameR, hCam - frameR), (0, hScr))
      curX = prevX + (x3 - prevX) / mouseSmooth
      curY = prevY + (y3 - prevY) / mouseSmooth
      mouse.position = (wScr-curX, curY)
      prevX, prevY = curX, curY
      cv2.circle(img, (x2, y2), 15, (255, 0, 255), -1)

    if fingers[1]==1 and fingers[2]==1:
      mouse.release(Button.left)

    if fingers[1] == 0 and fingers[2] == 1:
      mouse.press(Button.left)



  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

  cv2.imshow("Image", img)
  cv2.waitKey(1)