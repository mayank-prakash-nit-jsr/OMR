import cv2
import numpy as np
import utils

path="img_1.jpg"
widthImg=700
heightImg=700
img=cv2.imread(path)
            
img=cv2.resize(img,(widthImg,heightImg))
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)
#ret,threshold=cv2.threshold(imgBlur,127,255,0)
imgContours=img.copy()


contours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#print(len(contours))
#cv2.drawContours(imgContours,contours,-1,(0,255,0),5)
#cv2.imshow("Window",imgContours)

rectangleContour=utils.rectContours(contours)
biggestRectangleContour=rectangleContour[0];
#print(biggestRectangleContour)  # definitely will array of points not four only because it is locus of points. 
#print(utils.cornerPoints(biggestRectangleContour))  # Now it will give all four corner points of biggest rectangle.

MarkedAns=utils.cornerPoints(biggestRectangleContour) # Lets say MarkedAns of biggest rectangle four corner points.
GradePoints=utils.cornerPoints(rectangleContour[1])   # Next biggest rectangle four corner points lets say will be grade area in our image.
#print(GradePoints) 
#cv2.drawContours(imgContours,rectangleContour[0],-1,(0,255,0),5)  # It draw contours.
#cv2.imshow("Window",imgContours)  # It will show biggest rectangle contour only , Similarly for second biggest rectangle you can see [1].
#cv2.drawContours(imgContours,MarkedAns,-1,(0,255,0),20)  # It draw contours.
#cv2.imshow("Window",imgContours)  # It will show biggest rectangle contour corner points only , Similarly for second biggest rectangle you can see [1].

# Now for any given image we will start our further process only after finding these two rectangles corner points.
if MarkedAns.size!=0 and GradePoints.size!=0:
    MarkedAns=utils.reorder(MarkedAns)
    GradePoints=utils.reorder(GradePoints)

    # Now we do warp perspective
    pts1=np.float32(MarkedAns)
    pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])  #widthImg,heightImg defined above 700,700
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgwarpColorMarkedAns=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    #cv2.imshow("Window",imgwarpColorMarkedAns)  # Now finally this is cutted image of MarkedAns

    pTS1=np.float32(GradePoints)
    pTS2=np.float32([[0,0],[300,0],[0,100],[300,100]])
    matrixG=cv2.getPerspectiveTransform(pTS1,pTS2)
    imgwarpColorGradePoints=cv2.warpPerspective(img,matrixG,(300,100))
    #cv2.imshow("Window",imgwarpColorGradePoints)  # Now finally this is cutted image of GradePoints

    # We will apply threshold on imgwarpColorMarkedAns to get which bubble is colored.
    imgwarpGrayMarkedAns=cv2.cvtColor(imgwarpColorMarkedAns,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Window",imgwarpGrayMarkedAns)
    # Gray then Threshold
    imgwarpThresholdMarkedAns=cv2.threshold(imgwarpGrayMarkedAns,170,255,cv2.THRESH_BINARY_INV)[1]  # [1] for holding and showing for sometime
    # If 0 here then color reverse instead of cv2.THRESH_BINARY_INV
    #cv2.imshow("Window",imgwarpThresholdMarkedAns)

    # Now threshold marked ans image will be splitted into each separate bubbles.
    boxes=utils.splitBoxes(imgwarpThresholdMarkedAns)
    #cv2.imshow("Window",boxes[2])  # It will show 3rd box of 1st row
    # Now we will count non-zero pixel beacuse marked is in white to find which boxes are marked and which are not.
    #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2])) # It will show count of both 2nd which is marked and 3rd which is not marked. So see differences.
    
    # So Actually 5 rows means 5 questions and each question has 5 choices.
    questions=5
    choices=5
    row=0
    column=0
    ans=[1,2,0,1,4] # It is the correct answers we know. 

    pixelMatrix=np.zeros((questions,choices))
    for bubble in boxes:
        totalPixels=cv2.countNonZero(bubble)
        pixelMatrix[row][column]=totalPixels
        column+=1
        if column==choices:
            row+=1
            column=0
    #print(pixelMatrix)
    
    myIndex=[]
    for x in range(0,questions):
        arr=pixelMatrix[x]
        #print("arr",arr) gives each row values in an array
        myIndexVal=np.where(arr==np.amax(arr)) # max also work
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    #print(myIndex)
        
    # Now we will grade the marks
    grading=[]
    for x in range(0,questions):
        if ans[x]==myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)    
    #print(grading)
    score=(sum(grading)/questions)*100  # score percentage
    #print(score)

    # Displaying Answers and Grades on paper
    # Answers

    imgResult=utils.showAnswers(imgwarpColorMarkedAns,myIndex,grading,ans,questions,choices)
    #cv2.imshow("Window",imgResult)
    imgRawDrawing=np.zeros_like(imgwarpColorMarkedAns)
    imgRawDrawing=utils.showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
    #cv2.imshow("Window",imgRawDrawing)

    # Now we will do inverse warp perspective this imgRawDrawing on original image
    invMatrix=cv2.getPerspectiveTransform(pts2,pts1)
    imgFinalWithRaw=cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))
    #cv2.imshow("Window",imgFinalWithRaw)
    
    # Now we will combine this resultant Raw with Original image
    imgFinal=img.copy()
    imgFinal=cv2.addWeighted(imgFinal,1,imgFinalWithRaw,1,0)
    #cv2.imshow("Window",imgFinal)

    # Answers
    imgRawGrade=np.zeros_like(imgwarpColorGradePoints)
    cv2.putText(imgRawGrade,str(int(score))+"%",(40,90),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    #cv2.imshow("Window",imgRawGrade)
    # Now do inverse Perspective
    invMatrixG=cv2.getPerspectiveTransform(pTS2,pTS1)
    imgInvGradePoints=cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))
    imgFinal=cv2.addWeighted(imgFinal,1,imgInvGradePoints,1,0)
    cv2.imshow("Window",imgFinal)
    cv2.waitKey(0)
    

    ## END