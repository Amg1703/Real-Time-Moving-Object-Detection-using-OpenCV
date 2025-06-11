import cv2
import time
import imutils

cam = cv2.VideoCapture(0) # USING VIDEO CAPTURE FUNCTION AND THE PARAMETER 0 IS FOR WEBCAMS 
time.sleep(1) # 

firstimage=None
area=500

while True:
    _,img=cam.read()
    text="Normal" # AT ALL TIMES THE TEXT WILL DISPLAY NORMAL UNLESS MOVING OBJECT IS DETECTED
    resizeimage=imutils.resize(img, width=500) # RESZING THE IMAGE FROM CAMERA
    grayimage=cv2.cvtColor(resizeimage, cv2.COLOR_BGR2GRAY) # CONVERTING IMAGE TO GRAY SCALE IMAGE 
    gaussianblurimage=cv2.GaussianBlur(grayimage, (21,21), 0) # (21,21) because it should be odd  PERFORMING GAUSSIAN-BLUR
    if firstimage is None: # AS THE FIRST IMAGE WAS DECLARED NONE
        firstimage=gaussianblurimage  # THE FIRST IMAGE WHICH WILL BE THE BACKGROUND IMAGE IS ASSUMED AS THE GAUSSIAN BLUR IMAGE
        continue
    imagediff=cv2.absdiff(firstimage,gaussianblurimage) # WHEREIN WE ARE COMPARING THE FIRST FRAME WHICH IS THE BACKGROUND IMAGE AND EACH IMAGE THAT IS OBTAINED AFTER THAT 
    thresholdimage=cv2.threshold(imagediff,25,255,cv2.THRESH_BINARY)[1]  # WE ARE APPLYING DESIRED THRESHOLD OF 25 TO THAT IMAGE WITH MAXIMUM THRESHOLD OF 255
    dilateimage=cv2.dilate(thresholdimage, None, iterations=2) # NOW THAT IMAGE IS BLACK AND WHITE WE NEED FILL THE SMALL HOLES THAT ARE CONVERTED WRONGLY AND BY THICKENNING THE OBJECTS IN THE IMAGE WE CAN DO SO
    cnts=cv2.findContours(dilateimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # WE ARE FINDING SIMILAR PIXELS IN THE IMAGE
    cnts=imutils.grab_contours(cnts) # WE ARE JOINING ALL THE SIMILAR PIXELS TOGETHER 
    for c in cnts: 
        if cv2.contourArea(c)>area: # ONLY IF THE JOINED PIXEL AREA IS LESS THAN THAT OF THE AREA OF IMAGE THE LOOP CONTINUES ELSE WILL IT BE NORMAL
            break
        (x,y,w,h) = cv2.boundingRect(c) # USING BOUNDING RECTANGLE FUNCTION ON C WE ARE FINDING THE CONTOUR COORDINATES WHICH ARE THE COORDINATES OF THE MOVING OBJECT
        cv2.rectangle(resizeimage,(x,y),(x+w,y+h),(0,255,0),2) # PLOTTING A RECTANGLE AROUND THE OBJECT
        text="Moving object is detected"  # TO DISPLAY THE TEXT OF MOVING OBJECT DETECTED
    print(text)
    cv2.putText(resizeimage, text, (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)  # PUT TEXT FUNCTION IS OUTSIDE THE LOOP
    cv2.imshow('Recorded video', resizeimage)
    key=cv2.waitKey(1) & 0xFF # HERE WAIT KEY HAS THE PARAMETER 1 WHICH AND WITH 0x11 TO STORE THE KEY AS 11111
    if key==ord("q"): # ON PRESSING THE KEY Q IT GENERATES A BINARY CODE WHIHC IS 11111 WHICH BREAKS THE LOOP
        break

cam.release()
cv2.destroyAllWindows()
