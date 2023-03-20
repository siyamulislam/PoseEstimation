import cv2 
import numpy as mp
import time

cap= cv2.VideoCapture('ai-train/squad.mp4')
if (cap.isOpened() == False):
    print("Unable to read camera feed")
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
while True:
    ret,img= cap.read()
    if not ret:
        print("Can't receive img (stream end?). Exiting ...")
        break
    cv2.imshow("image",img)
    key= cv2.waitKey(1)

    # press q key to close the program
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
