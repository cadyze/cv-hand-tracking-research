import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConf=0.5, trackingConf=0.5) -> None:
        self.mode=mode
        self.max_num_hands=maxHands
        self.complexity=complexity
        self.min_detection_confidence=detectionConf
        self.min_tracking_confidence=trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_num_hands, self.complexity, 
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        # Getting Landmarks
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList



# print(results.multi_hand_landmarks)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    while True:
        success, img = cap.read()
        detector.findHands(img)
        detector.findPosition(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()