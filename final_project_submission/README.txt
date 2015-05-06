Solution for CS-8803-O01 Final Project
======================================
Last Revised Aug 8th, 2014

Team Members:
- Yancheng (Andy) Liu
- Umashankar Gaddameedi

-------------------------
1. INTRODUCTION
This is a Python program that predicts the coordinates of the centroid of the hexbug for the next ~2.5 seconds beyond the end point of the video data.  The program takes the hexbug-testing_video.mp4 or another hexbug video as an input.  The output will be a text file, named prediction.txt, containing a list of 63 pairs of integers.  Each pair of the integers represents one frame of the video.  The first element of the each pair is the x-coordinate of the centroid and the second element is the y-coordinate.

The program uses a combination of an autoregressive model and circular fitting to predict the future positions of the moving hexbug.  A reflection model is used to calculate the heading direction change upon hitting the boundaries.  Training video is used to calculate best estimation of turning angle and tangent velocity.  The autoregressive model is used as described by Ashraf Elnagar (Intelligent Control and Automation, 2011, 2, 284-292), and the coefficients are estimated using the conditional maximum likelihood approach.   

2. DEPENDENCIES
To run the program one will need Python 2.7.8, opencv 2.4.9, and numpy 1.8.1.

3. RUNNING THE PROGRAM
- Open the hexbug_prediction.py by IDLE
- In the __name__ == '__main__' section, modify the code to specify the paths for 
	a, input files (hexbug-training_video-transcoded.mp4 and hexbug-testing_video.mp4) 
	b, files for saving the extracted centroid data
	c, output file (prediction.txt) 
  * Note: to take a different video as input, replace the file path and name of hexbug-testing_video.mp4 with the file path and name of the new video. 
- Run the module in Python Shell
- The output file containing the prediction data will be saved at user specified location.

4. PERFORMANCE TEST
The performance of the program is tested across each 63-frame segment in the testing_video.  With a total of over 1200 tests, the accuracy of the prediction is summarized as follows (L^2 error):
 - Min: 342
 - Max: 2299
 - Average: 1174 
 - StdDev: 332  

