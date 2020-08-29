import cv2
import numpy as np

##################################################################################

# Finding corner points of biggest rectangle
# We filter small small area which we assume 50 which is very small by seeing our image 

def rectContours(contours):
    rectConts=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>50:
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True) # 0.02 is resolution , This function gives corner points.
            #print("Corner Points ",len(approx))
            if len(approx)==4:
                rectConts.append(cnt)
    rectConts = sorted(rectConts, key=cv2.contourArea,reverse=True)  # Now I sort all rectangles in decreasing order so that 1st is biggest rectangle on area basis.
    return rectConts

########################################################################################################################################

# Now we find four points of biggest rectangle

def cornerPoints(biggestRectangleContour):
     peri=cv2.arcLength(biggestRectangleContour,True)
     approx=cv2.approxPolyDP(biggestRectangleContour,0.02*peri,True) # 0.02 is resolution , This function gives corner points.
     return approx

#########################################################################################################################################

# Now we reorder points which is 1st,2nd etc. to get warp perpespective i.e Bird Eye View

def reorder(myPoints):
    #print(np.shape(myPoints))
    myPoints=myPoints.reshape((4,2))
    myNewPoints=np.zeros((4,1,2),np.int32)
    add=np.sum(myPoints,axis=1) # Add all x,y coordinates and give results in 4 rows. Lowest among them will become origin. Highest its diagonal.Think about screen.
    #print(myPoints)
    #print(myNewPoints)
    #print(add)
    myNewPoints[0]=myPoints[np.argmin(add)] #[0,0]
    myNewPoints[3]=myPoints[np.argmax(add)] #[w,h]  index we decide by seeing add
    # Now Take advantage of diff
    diff=np.diff(myPoints,axis=1)
    myNewPoints[1]=myPoints[np.argmin(diff)] #[w,0]
    myNewPoints[2]=myPoints[np.argmax(diff)] # [0,h]
    #print(diff)
    return myNewPoints

#####################################################################################################

# Now we split the threshold marked ans image into five rows according to structure of image and then each row in 5 columns so that to get all 25 bubbles.
def splitBoxes(img):
    rows=np.vsplit(img,5) # It will cut image in 5 equal area row wise.
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

####################################################################################################   

# Showing grades and answers on image
def showAnswers(img,myIndex,grading,ans,questions,choices):
    sectionWidth=int(img.shape[1]/questions)
    sectionHeight=int(img.shape[0]/choices)
    for x in range(0,questions):
        myAns=myIndex[x]
        centreX=int((myAns*sectionWidth)+sectionWidth/2)
        centreY=int((x*sectionHeight)+sectionHeight/2)
        if grading[x]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)    
            correctAns=ans[x]
            cv2.circle(img,(int((correctAns*sectionWidth)+sectionWidth/2),int((x*sectionHeight)+sectionHeight/2)),50,(255,0,0),cv2.FILLED) # Which is correct in blue if not marked correct
        cv2.circle(img,(centreX,centreY),50,myColor,cv2.FILLED)
    return img

####################################################################################################

