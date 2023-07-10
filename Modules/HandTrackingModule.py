import cv2
import mediapipe as mp
import time
import math
import numpy as np
import applescript

class handDetector():
  def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils

    # id's of each finger tip going from thumb to pinky.
    self.tipIds = [4, 8, 12, 16, 20]


  
  def findHands(self, img, draw=True, flipType=True, hand="Both"):
    # returns: a list of Hands with handedness, img with labels if draw

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    allHands = []
    h, w, c = img.shape

    if self.results.multi_hand_landmarks:
      for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
        myHand = {}
        mylmList = []
        xList = []
        yList = []
        for id, lm in enumerate(handLms.landmark):
           px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
           mylmList.append([px, py, pz])
           xList.append(px)
           yList.append(py)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH

        if flipType:
          if handType.classification[0].label == "Right":
            myHand["type"] = "Left"
          else:
            myHand["type"] = "Right"
        else:
          myHand["type"] = handType.classification[0].label
        allHands.append(myHand)
        
        if draw:
          self.mpDraw.draw_landmarks(img, handLms,
              self.mpHands.HAND_CONNECTIONS)
          cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), 
                        (255, 0, 255), 2)
          cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                      2, (255, 0, 255), 2)
    print(allHands)
    if draw:
      return allHands, img
    else:
      return allHands
    

  def findPosition(self, img, allHands, hand="Right", handNo=0, draw= False, flipType= True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []

    if self.results.multi_hand_landmarks:
      # Switches HandNo depending on ordering of lmList
      if(hand != "Both" and allHands[0]["type"] != hand):
        if (len(allHands) == 1):
          return [], []
        else:
          handNo = 1
      myHand = self.results.multi_hand_landmarks[handNo]
      
      for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        xList.append(cx)
        yList.append(cy)
        self.lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
      xmin, xmax = min(xList), max(xList)
      ymin, ymax = min(yList), max(yList)
      bbox = xmin, ymin, xmax, ymax

      if draw:
        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
    return self.lmList, bbox

  # returns an array of length 5, with 0 index representing the thumb, and 4 index representing the pinky.
  def fingersUp(self):
    fingers = []
    # Thumb
    if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
      fingers.append(1)
    else:
      fingers.append(0)

    # Fingers
    for id in range(1, 5):
      if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
        fingers.append(1)
      else:
        fingers.append(0)

      # totalFingers = fingers.count(1)

    return fingers

  def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
      x1, y1 = self.lmList[p1][1:]
      x2, y2 = self.lmList[p2][1:]
      cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

      if draw:
          cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
          cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
          cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
          cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
          length = math.hypot(x2 - x1, y2 - y1)

      return length, img, [x1, y1, x2, y2, cx, cy]
  
  def setVolume(self, volume):
    applescript.AppleScript(f'set volume output volume {volume}').run()

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img, flipType=False)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()