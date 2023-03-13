import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose=mp.solutions.pose;
pose = mpPose.Pose()
pTime = 0; cTime = 0;
cap = cv2.VideoCapture('PoseVideos/3.mp4')
# cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

while (True):
    ret, img = cap.read()
    if not ret:
        print("Can't receive img (stream end?). Exiting ...")
        break
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 60),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    bg=cv2.imread('bg2.png')

    if results.pose_landmarks:
        mpDraw.draw_landmarks(bg,results.pose_landmarks,mpPose.POSE_CONNECTIONS,
        mpDraw.DrawingSpec(color=(0, 0, 255),),mpDraw.DrawingSpec(color=(0, 255, 0), ))
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS,
        mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(id,cx,cy)


    k = cv2.waitKey(1)
    cv2.imshow('Pose Estimation4', img)
    cv2.imshow('Pose Estimation', bg)

    # press q key to close the program
    if k & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
