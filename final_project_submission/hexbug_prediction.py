import cv2
import numpy as np
import math


def hexbug_tracker(filePath, targetFilePath):
    '''the hexbug_tracker method is provided in the original assignment
    it converts hexbug movement in a video into a point position 
    '''
    cap = cv2.VideoCapture(filePath)
    #TODO fix a filePath to save the converted txt file
    fout = open(targetFilePath,'w')
    fout.write('[')
    
    centroid = []
    ctr = 0
    
    hmin = (238-30)/2
    hmax = (238+30)/2
    
    smin = 255/2
    smax = 255
    
    vmin = int(0.20 * 255)
    vmax = int(1.0 * 255)
    
    # define range of color in HSV
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    
    while True:
    
        try:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ctr += 1
            if ctr % 100 == 0:
                print 'frame', ctr
                s = ''
                for c in centroid:
                    s += str(c) + ',\n'
                fout.write(s)
                del centroid[:]
        except:
            break
  
        mask = cv2.inRange(hsv, lower, upper)
    #    cv2.imshow('mask',mask)
    
        ret,thresh = cv2.threshold(mask,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centroid.append([cx,cy])
            else:
                centroid.append([-1,-1])
     
    s = ''
    for c in centroid:
        if c is centroid[-2]:
            s += str(c) + ']\n'
            break
        else:
            s += str(c) + ',\n'
    fout.write(s)
    
    #print 'all done!'
    
    fout.close()
    cap.release()
    cv2.destroyAllWindows()

def readData(fo):
    '''the readData method reads all centroid position data 
    and returns a list containing these data for following calculation
    '''
    lines = fo.readlines()
    pointData =[]
    for i in range(len(lines)):
        line = lines[i]
        if line[1] == '-':      # remove all invalid measurement (-1, -1))
            continue
        start = 1
        if i == 0:
            start =2            # the very first line contains an extra '['
        # parsing the string containing X position till ','
        commaIndex = line.find(',')     
        x = int(line[start: commaIndex])# converting to number 
        # parsing the string containing y position till ']'
        finishIndex = line.find(']', commaIndex) 
        y = int(line[commaIndex+1:finishIndex])
        pointData.append((x,y))    
    return pointData

def getBoundary(pointData):
    '''the getBoundary method calculate the area within which the hexbug is able to reach.
    It returns the positions of boundaries in the sequence of: 
    left_wall, ceiling, right_wall, floor
    '''
    #initiate with values far out of the moving area
    minX = 999 
    minY = 999
    maxX = 0
    maxY = 0
    for i in range (len(pointData)):
        if pointData[i][0] < minX:
            minX = pointData[i][0]
        if pointData[i][0] > maxX:
            maxX = pointData[i][0]
        if pointData[i][1] < minY:
            minY = pointData[i][1]
        if pointData[i][1] > maxY:
            maxY = pointData[i][1]
    buffering_distance = 15 
    # the buffering_distance is contributed by
    #1. the hexbug has dimension and can turn before hitting the wall
    #2. the hexbug may turn between two frames of the movie
    #thus the boundary is shrunk by a buffering_distance
    # the value is optimized by minimizing the L^2 value with training data
    return (minX+buffering_distance, minY+buffering_distance, maxX-buffering_distance, maxY-buffering_distance)
    
def calculate_velocity_and_angle(pointData, worldSize):
    '''One model to predict the hexbug movement is to assume that it is moving in a circular motion.
    Thus it should have a fixed steering angle and mostly has a constant tangent velocity.
    
    The calculate_velocity_and_angle method uses data from the training video to calculate and return the best estimation 
    for steering angle and tangent velocity, which will be used for prediction    
    '''
    #calculate tangent velocities
    tangentVelocities = []
    
    for i in range (len(pointData)-1):
        edge = 5 # excluding points that are too close to boundaries
        if pointData[i+1][0]< worldSize[0]+edge or pointData[i+1][0]> worldSize[2]-edge or \
            pointData[i+1][1]< worldSize[1]+edge or pointData[i+1][1] > worldSize[3] -edge:
            continue
        # calculate distance moved in unit time (one frame)
        distance = math.sqrt((pointData[i+1][0]-pointData[i][0])**2 + (pointData[i+1][1]-pointData[i][1])**2)
        tangentVelocities.append(distance) # append distance to a list
    
    #calculate steering angles
    steeringAngle = []    
    for i in range (len(pointData)-2):
        edge = 10 # excluding points that are too close to boundaries
        if pointData[i+1][0]< worldSize[0]+edge or pointData[i+1][0]> worldSize[2]-edge or \
            pointData[i+1][1]< worldSize[1]+edge or pointData[i+1][1] > worldSize[3] -edge:
            continue
        #calculate the change of heading angle
        angle1 = math.atan2(pointData[i+2][1]-pointData[i+1][1], pointData[i+2][0]-pointData[i+1][0])
        angle2 = math.atan2(pointData[i+1][1]-pointData[i][1], pointData[i+1][0]-pointData[i][0])
        # remove outliners due to randomness or sudden jump of angle differences
        if angle2-angle1 < 0.5 and angle2-angle1 > -0.5 :
            steeringAngle.append(angle2-angle1)
    
    #calculate the average and standard deviation of tangent velocities
    velocityMean = 0
    for i in range (len(tangentVelocities)):
        velocityMean += tangentVelocities[i]
    velocityMean = velocityMean/ len(tangentVelocities)
    velocityStd = 0
    for i in range (len(tangentVelocities)):
        velocityStd += (tangentVelocities[i]-velocityMean)**2
    velocityStd = math.sqrt(velocityStd/len(tangentVelocities))
    
    ##calculate the average and standard deviation of steering angles
    angleMean = 0
    for i in range (len(steeringAngle)):
        angleMean += steeringAngle[i]
    angleMean = angleMean/ len(steeringAngle)
    AngleStd = 0
    for i in range (len(steeringAngle)):
        AngleStd += (steeringAngle[i]-angleMean)**2
    AngleStd = math.sqrt(AngleStd/len(steeringAngle))
    
    return velocityMean, angleMean
    
def predict_next_by_circlular_fitting(pointData, current, velocityMean, angleMean, collision):
    '''One model to predict the hexbug movement is to assume that it is moving in a circular motion.
    Thus it should have a fixed steering angle and mostly has a constant tangent velocity.(assume acceleration is zero)
    
    The predict_next_by_circlular_fitting method uses the best estimation of steering angle and tangent velocity calculated 
    in calculate_velocity_and_angle method, namely, velocityMean and angleMean to predict the position of the next time point. 
    It has better performance for time points closer to known positions than autoregressive model 
    
    pointData is a list contains last 3 know points for calculating the latest position and heading 
    angle and will be appended with new estimated positions
    
    current tracks the last points in the list 
    
    collision records whether hexbug hit wall the last time predict_next_by_circlular_fitting method was called and whether it bounces vertically or horizontally'''
    
    if collision == 0: # did not hit any wall
        #update heading angle of next time point
        headingAngle = math.atan2(pointData[current][1]-pointData[current-1][1], pointData[current][0]-pointData[current-1][0])
    if collision == 1: # hit left or right wall
        #mirroring the heading direction horizontally back to the frame before collision 
        headingAngle = math.pi-math.atan2(pointData[current-1][1]-pointData[current-2][1], pointData[current-1][0]-pointData[current-2][0])
        #then add corresponding steering angles changed in this time frame
        headingAngle += angleMean   
        collision =0
    if collision == 2: # hit ceiling or bottom
        #mirroring the heading direction vertically back to the frame before collision 
        headingAngle = 2*math.pi - math.atan2(pointData[current-1][1]-pointData[current-2][1], pointData[current-1][0]-pointData[current-2][0])
        headingAngle += angleMean
        collision =0
    
    headingAngle += angleMean
    #predict the next position of hexbug and examine whether it will hit any wall before reaching there
    posX = pointData[current][0] + velocityMean*math.cos(headingAngle)
    
    if posX < worldSize[0]:
                posX = worldSize[0]*2 - posX
                collision = 1
    if posX > worldSize[2]:
                posX = worldSize[2]*2 - posX
                collision = 1
                
    posY = pointData[current][1] + velocityMean*math.sin(headingAngle)
    
    if posY < worldSize[1]:
                posY = worldSize[1]*2 - posY
                collision = 2
    if posY > worldSize[3]:
                posY = worldSize[3]*2 - posY
                collision = 2
    return (posX, posY), collision

def predict_next_by_ARM(pointDataOriginal, n =6):
    '''This method uses autoregressive model (ARM) to predict the hexbug movement.  The model assumes that the change of acceleration is slow and steady.
    Please see README.txt files for detailed explanation and references. 
    
    This model needs at least 4 points to predict future trajectory. It dose 
    not assume acceleration is zero and has better performance than the circular model in longer range prediction.'''
    
    current = len(pointDataOriginal)-1 # the last point before prediction, must be bigger than n
    
    pointData = []
    for i in range(current+1):
        pointData.append([pointDataOriginal[i][0],pointDataOriginal[i][1]])
  
    # list of accelerations at different points in X direction, most recent point comes first
    X_accelerations = [] 
    Y_accelerations = []
    
    for i in range(n-2):
        X_accelerations.append(pointData[current-i][0]- 2*pointData[current-1-i][0]+ pointData[current-2-i][0])
        Y_accelerations.append(pointData[current-i][1]- 2*pointData[current-1-i][1]+ pointData[current-2-i][1])
   
    #calculate beta values to estimation, please see equations in reference 
    numerator = 0
    denominator = 0
    for i in range(n-3):
        numerator += X_accelerations[i]*X_accelerations[i+1]
        denominator += X_accelerations[i+1]**2 
    betaX = 1.0*numerator/denominator
    
    numerator =0
    denominator =0
    for i in range(n-3):
        numerator += Y_accelerations[i]*Y_accelerations[i+1]
        denominator += Y_accelerations[i+1]**2
    betaY = 1.0*numerator/denominator
    
    nextPoints_FittingMode = []
    for i in range(63):
        nextX = (2+betaX)*pointData[current][0]-(1+2*betaX)*pointData[current-1][0]+ betaX*pointData[current-2][0]
        nextY = (2+betaY)*pointData[current][1]-(1+2*betaY)*pointData[current-1][1]+ betaY*pointData[current-2][1]
        # add new estimation in list for next estimation
        pointData.append((nextX, nextY)) 
        nextPoints_FittingMode.append([nextX, nextY])  
        current += 1   
    return nextPoints_FittingMode


def predict_next_combined(worldSize, velocity, angle, pointData_testing):
    '''combined two models for best estimation'''
    
    roundOfPrediction = 1 #  During testing stage, can be set up to 1200 to estimate known frames and calculate L^2 based on measured data
    
    for i in range(roundOfPrediction):
        
        #Prediction with circular model
        
        PointTostartPrediction = len(pointData_testing) #During testing, PointTostartPrediction can be set as i + 10
        collision = 0
        nextPoints_CircularMode = []
        #read in 3 known points to initiate estimation
        for i in range(PointTostartPrediction - 3, PointTostartPrediction):
            nextPoints_CircularMode.append(pointData_testing[i])
        
        for i in range(63):
            nextPoint, collision = predict_next_by_circlular_fitting(nextPoints_CircularMode, i + 2, velocity, angle, collision)# start estimation from last know points: i+2
            nextPoints_CircularMode.append(nextPoint)
        
        nextPoints_CircularMode = nextPoints_CircularMode[3:] # trim off first three known points
        
        #Prediction with AR model (without considering collision)
        
        nextPoints_ARM = predict_next_by_ARM(pointData_testing)
        
        #deal with collision and boundary
        for i in range(len(nextPoints_ARM)):
            while nextPoints_ARM[i][0] < worldSize[0] or nextPoints_ARM[i][0] > worldSize[2]:
                if nextPoints_ARM[i][0] < worldSize[0]:
                    nextPoints_ARM[i][0] = worldSize[0] * 2 - nextPoints_ARM[i][0]
                if nextPoints_ARM[i][0] > worldSize[2]:
                    nextPoints_ARM[i][0] = worldSize[2] * 2 - nextPoints_ARM[i][0]
            
            while nextPoints_ARM[i][1] < worldSize[1] or nextPoints_ARM[i][1] > worldSize[3]:
                if nextPoints_ARM[i][1] < worldSize[1]:
                    nextPoints_ARM[i][1] = worldSize[1] * 2 - nextPoints_ARM[i][1]
                if nextPoints_ARM[i][1] > worldSize[3]:
                    nextPoints_ARM[i][1] = worldSize[3] * 2 - nextPoints_ARM[i][1]
        
        #combine this two models: prefer circular model at the beginning and Fitting model later on
        next63Points = []
        for i in range(63):
            ratio = 1.0 / 63 * i
            xPos = ratio * nextPoints_ARM[i][0] + (1 - ratio) * nextPoints_CircularMode[i][0]
            yPos = ratio * nextPoints_ARM[i][1] + (1 - ratio) * nextPoints_CircularMode[i][1]
            next63Points.append([xPos,yPos])
    return next63Points


def outputTXT(next63Points, fo3):
    fo3.write('[')
    for i in range(len(next63Points)):
        next63Points[i][0] = int(round(next63Points[i][0]))
        next63Points[i][1] = int(round(next63Points[i][1]))
    
    s = ''
    for i in range(62):
        s += str(next63Points[i]) + ',' + '\n'
        fo3.write(s)
        s = ''
    
    s += str(next63Points[62]) + ']'
    fo3.write(s)
    print 'all done!'
    
def calculateL2(realPoints, EstimatedPoints, current):
    L_square = 0
    
    for i in range(63):
        L_square += (realPoints[current + i + 1][0] - EstimatedPoints[i][0]) ** 2 + (realPoints[current + i + 1][1] - EstimatedPoints[i][1]) ** 2
    
    print math.sqrt(L_square)


if __name__ == '__main__':
 
 #################################################################################################
    
    file_training_video = 'C:/CS8803/FinalProject/hexbug-training_video-transcoded.mp4' # edit to specify the file path for hexbug-training_video-transcoded.mp4
    file_training_centroid_data = 'C:/CS8803/FinalProject/training_centroid' # edit to specify the file path for saving training_centroid data extracted from the video
    file_testing_video = 'C:/CS8803/FinalProject/hexbug-testing_video.mp4' # edit to specify the file path for hexbug-testing_video.mp4
    file_testing_centroid_data = 'C:/CS8803/FinalProject/testing_centroid' # edit to specify the file path for saving testing_centroid data extracted from the video
    output_file = 'C:/CS8803/FinalProject/Prediction.txt' # edit to specify the file path for the output Prediction.txt
 
 #################################################################################################
   
    hexbug_tracker(file_training_video, file_training_centroid_data)
    hexbug_tracker(file_testing_video, file_testing_centroid_data)
    
    fo1 = open(file_training_centroid_data)
    pointData_training = readData(fo1)
    fo2= open(file_testing_centroid_data)
    pointData_testing = readData(fo2)
    
    worldSize = getBoundary(pointData_training)
    velocity, angle = calculate_velocity_and_angle(pointData_training, worldSize)
    
    next63Points = predict_next_combined(worldSize, velocity, angle, pointData_testing)
        
    fo3 = open(output_file,'w')
    outputTXT(next63Points, fo3)
                 
    fo1.close()
    fo2.close()
    fo3.close()
