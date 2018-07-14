import cv2
import numpy as np


imgW = 340
imgH = 220
gridH = imgH//3
gridW = imgW//4

last = (0,0) # track last position

alpha = .7

icons = cv2.imread('icons.png',0) # read in backgroun image
iconsChannels = np.zeros((imgH,imgW,3),np.uint8) # initialize np array

# invert the icons image
for row in range(len(icons)):
    for col in range(len(icons[0])):
        icons[row][col] = 255 if icons[row][col] == 0 else 0
black = np.where(icons > 100)

# convert sing channel to three channels
for row_i in range(len(icons)):
    for col_i in range(len(icons[0])):
        iconsChannels[row_i][col_i] = (icons[row_i][col_i],icons[row_i][col_i],icons[row_i][col_i])

# get which square on the grid the x,y position falls in
def getGrid(x,y):
    return (x//gridW)*gridW, (y//gridH)*gridH

def trackFinger():
    global last

    # bound on green
    lowerBound=np.array([40,80,40])
    upperBound=np.array([102,255,255])

    cam = cv2.VideoCapture(0)
    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))

    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)

    while True:
        canvas = np.zeros((imgH,imgW,3),np.uint8) # initialize canvas
        canvas[:]  = (192,192,192) # set the background to gray
        a,img = cam.read()
        img=cv2.resize(img,(imgW,imgH))
        img=cv2.flip(img,1)

        # convert image bgr to hsv
        imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

        maskFinal=maskClose
        conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        leftX, topY = getGrid(last[0],last[1])
        cv2.drawContours(img,conts,-1,(0,255,0),3) # draw a bounding box around found color

        # for simplicity, just randomly select the first contour
        if len(conts) != 0: # check that target color was detected
            x,y,w,h=cv2.boundingRect(conts[0])
            moments = cv2.moments(conts[0])
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                last = (cX,cY) # store this as the last location
            else:
                # if nothing found, use the last location for stability
                cX = last[0]
                cY = last[1]
            leftX, topY = getGrid(cX,cY)
            last = (leftX, topY) # store last location
            overlay = img.copy()

            cv2.circle(img,(cX,cY),(w+h)/4,(0,0,255), 2) # draw a circle around the center of green location

        cv2.rectangle(canvas,(leftX,topY),(leftX + gridW, topY + gridH),(255,255,255), -2)

        canvas[black[0],black[1]] = (0,0,0) # draw icons onto canvas

        # show camera for debugging
        # cv2.namedWindow("cam")
        # cv2.imshow("cam",img)

        cv2.namedWindow("canvas")
        cv2.imshow("canvas",canvas)

        cv2.waitKey(10)

trackFinger()
