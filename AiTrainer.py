import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture('ai-train/hand-work.mp4')
if (cap.isOpened() == False):
    print("Unable to read camera feed")
wCam, hCam = 640, 480
count =0; dir= 0;pTime = 0;
detector=pm.poseDetector()
while True:
    ret, img = cap.read()
    if not ret:
        print("Can't receive img (stream end?). Exiting ...")
        break
    img= cv2.resize(img,(1280,720))

    # img= cv2.imread('ai-train/person.png')
    # img=cv2.flip(img,1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (10, 60),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    lmList =detector.findPosition(img,draw=False)
    detector.findPose(img , draw=False)
    # rightArm
    detector.findAngle(img,12,14,16)
    # leftArm
    angle= detector.findAngle(img,11,13,15)
    per =np.interp(angle,(210,320),(0,100))
    # print(per)

    # check for the dumble curl
    if per ==100:
        if dir ==0:
            count+=0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count+=0.5
            dir =0
    # print(count)
    cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
    cv2.putText(img,f'{int(count)}',(45,670),cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),25)


    cv2.imshow("image", img)
    key = cv2.waitKey(1)

    # press q key to close the program
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
