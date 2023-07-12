import math
import cv2

class HandVolume():
    def __init__(self, detector):
        self.detector = detector
    
    def volumeControl(self, img, lmList, bbox, handType):
        if len(lmList[handType]) > 0:
        # print(lmList[4], lmList[8])

            x1, y1 = lmList[handType][4][1], lmList[handType][4][2]
            x2, y2 = lmList[handType][8][1], lmList[handType][8][2]
            cv2.circle(img, (x1, y1), 15, (255,0,0), -1)
            cv2.circle(img, (x2, y2), 15, (255,0,0), -1)
            cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 15, (255,0,0), -1)
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)

            length = math.hypot(x2-x1, y2-y1)
            # print(length)
            bboxHeight = bbox[3] - bbox[1]
            vol = ((length) / bboxHeight) * 100
            self.detector.setVolume(vol)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, 400 - int(2.5 * int(vol))), (85, 400), (255, 0, 0), -1)
            cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        return img
    
    def volumeSignal(self, lmList, bbox, handType):
        if (len(lmList[handType]) > 0):
            fingers = self.detector.fingersUp(handType)
            return fingers == [1, 1, 0, 0, 0]
        return False

