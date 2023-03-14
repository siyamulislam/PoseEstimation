import cv2
import mediapipe as mp
import time


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

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        bg = cv2.imread('bg2.png')
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(bg, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 255), ),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), ))
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(
                                               color=(0, 0, 255), thickness=2, circle_radius=2),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        return img

    def findPosition(self, img, draw=True,drawID=0):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id,cx,cy)
            if id == drawID:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


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
        img = detector.findPose(img)
        detector.findPosition(img,drawID=28)

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
