# Moving over the HandMouse.py to a module file, in progress.
import numpy as np
import math
from screeninfo import get_monitors
from pynput.mouse import Button, Controller


class HandMouse():


  def __init__(self, detector):
    self.detector = detector
    self.mouse = Controller()
    self.wScr, self.hScr = get_monitors()[0].width, get_monitors()[0].height
    self.frameR = 100
    self.mouseSmooth = 5
    self.prevX = 0
    self.prevY = 0
    self.curX = 0
    self.curY = 0
    self.wCam = 640
    self.hCam = 640
    self.secondary = "Left"


  def mouseControl(self, lmList, handType):
    if len(lmList[handType]) != 0:
      x1, y1 = lmList[handType][8][1:]
      x2, y2 = lmList[handType][12][1:]      
      # print(fingers)


      x3 = np.interp(x2, (self.frameR, self.wCam - self.frameR),(0, self.wScr))
      y3 = np.interp(y2, (self.frameR, self.hCam - self.frameR), (0, self.hScr))

      self.curX = self.prevX + (x3 - self.prevX) / self.mouseSmooth
      self.curY = self.prevY + (y3 - self.prevY) / self.mouseSmooth
      self.mouse.position = (self.wScr-self.curX, self.curY)
      self.prevX, self.prevY = self.curX, self.curY

      # Defining the other secondary hand as clickFingers
      if (len(lmList["Left"])):
        cx1, cy1 = lmList[self.secondary][4][1], lmList[self.secondary][4][2]
        cx2, cy2 = lmList[self.secondary][8][1], lmList[self.secondary][8][2]
        length = math.hypot(cx2-cx1, cy2-cy1)
        print(length)
        if length < 13:
          self.mouse.press(Button.left)
        else:
          self.mouse.release(Button.left)

  def mouseSignal(self,lmList, handType="Right"):
    if (len(lmList[handType]) > 0):
      fingers = self.detector.fingersUp(handType)
      return fingers == [0, 1, 1, 0, 0]
    return False
