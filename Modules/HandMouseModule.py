# Moving over the HandMouse.py to a module file, in progress.

class HandMouse():
  def __init__(self, detector):
    self.detector = detector


  def handMouse(self):

    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam - frameR), (255, 0, 255), 2)

    if len(lmList) != 0:
      x1, y1 = lmList[8][1:]
      x2, y2 = lmList[12][1:]
      fingers = detector.fingersUp()
      # print(fingers)

    if fingers[1]==1 and fingers[2]==1:
      x3, y3 = np.interp(x2, (frameR, wCam - frameR), (0, wScr)), np.interp(y2, (frameR, hCam - frameR), (0, hScr))

      curX = prevX + (x3 - prevX) / mouseSmooth
      curY = prevY + (y3 - prevY) / mouseSmooth
      mouse.position = (wScr-curX, curY)
      prevX, prevY = curX, curY
      cv2.circle(img, (x2, y2), 15, (255, 0, 255), -1)

    if fingers[1] == 0 and fingers[2] == 1:
      mouse.press(Button.left)
      mouse.release(Button.left)