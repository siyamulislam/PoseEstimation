import cv2
import mediapipe as mp
import time
import math


class poseDetector():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable_sgm=False, smooth_sgm=True,
                 detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks,
        self.enable_sgm = enable_sgm,
        self.smooth_sgm = smooth_sgm,
        self.detection_con = detection_con,
        self.tracking_con = tracking_con,

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose((self.mode, self.complexity, self.smooth_landmarks, self.enable_sgm,
                                      self.smooth_sgm, self.detection_con, self.tracking_con))



    def findPosition(self, img, draw=True,drawID=0):
        self.lmList=[]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    if id == drawID:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return self.lmList
    def findAngle(self, img,p1,p2,p3,draw=True):
        #get the landmark
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]

        #calculate the Angel
        # angle=math.degrees(math.atan2(y1-y2,x1-x2)-math.atan2(y3-y2,x3-x2))

        angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        if angle<0:
            angle+=360
        # print(angle)
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x2,y2),(x3,y3),(255,255,255),3) 

            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255),2) 
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255),2) 
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255),2) 
            # cv2.putText(img, str(int(angle)),(x2-40,y2),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        return angle


    def findPose(self, img, draw=True):
        bg = cv2.imread('bg2.png')
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(bg, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 255), ),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), ))
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(
                                               color=(0, 0, 255), thickness=2, circle_radius=2),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        return img

def main():
    pTime = 0
    cap = cv2.VideoCapture('PoseVideos/3.mp4')
    # cap = cv2.VideoCapture(0)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    detector = poseDetector()
    while (True):
        ret, img = cap.read()
        if not ret:
            print("Can't receive img (stream end?). Exiting ...")
            break
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        detector.findPosition(img,drawID=23,draw=False)
        img = detector.findPose(img)

        k = cv2.waitKey(1)
        cv2.imshow('Pose Estimation4', img)
        # cv2.imshow('Pose Estimation', bg)

        # press q key to close the program
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
