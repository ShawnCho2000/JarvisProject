import math

class HandVolume():
    def __init__(self, detector):
        self.detector = detector
    
    def volumeControl(self, lmList, bbox, handType):
        if len(lmList[handType]) > 0:
        # print(lmList[4], lmList[8])

            x1, y1 = lmList[handType][4][1], lmList[handType][4][2]
            x2, y2 = lmList[handType][8][1], lmList[handType][8][2]

            length = math.hypot(x2-x1, y2-y1)
            # print(length)
            bboxHeight = bbox[3] - bbox[1]
            vol = ((length) / bboxHeight) * 100
            self.detector.setVolume(vol)
    
    def volumeSignal(self, lmList, bbox, handType):
        if (len(lmList[handType]) > 0):
            fingers = self.detector.fingersUp(handType)
            return fingers == [1, 1, 0, 0, 0]
        return False

