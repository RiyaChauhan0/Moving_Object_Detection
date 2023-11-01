import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500

while True:
    _,img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg,(21,21),0)
    if firstFrame is None:
        firstFrame = blurImg
        continue
    imgDiff = cv2.absdiff(firstFrame, blurImg)
    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    contourImg = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourImg = imutils.grab_contours(contourImg)
    for c in contourImg:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
        text = "Moving Object Detected"
        print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv2.imshow("CameraFeed",img)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break
cam.release()
cv2.destroAllWindows()
