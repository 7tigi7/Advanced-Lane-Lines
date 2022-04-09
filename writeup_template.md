## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

The code for all step is contained in the IPython notebook located in "./Project2.ipynb". 

### Dependencies

Lets start with some important dependencies to this project

`
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
import os
import glob
from PIL import Image
%matplotlib inline
`
### PLEASE NOTE THAT
Every used function and class can be found under title "Main functions, classes" at the start of the project.


### Camera Calibration

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Read more about distortion:
[Distortion](https://en.wikipedia.org/wiki/Distortion_(optics))

[Undistorted]: ./examples/undistort.png "Camera Calibration"
`
# This function performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
`

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
`
testImgToDistort = plt.imread('./test_images/test1.jpg')
cal_undistort(testImgToDistort,objpoints,imgpoints)
showOriginalAndUndistortedImage(testImgToDistort,objpoints,imgpoints)
`
[Undistorted Original]: ./examples/undistort_on_original.png "Camera Calibration"

### 2. Color transforms, gradients to create a thresholded binary image.
#### Can we found a better solution to find lanes on grayscaled or other color transformed images ?
I used a combination of color and gradient thresholds to generate a binary image.
Here's an example of my output for this step. 
`

testImgToColorTransform = plt.imread('./test_images/test2.jpg')

thresHold = (180, 255)
gray = cv2.cvtColor(testImgToColorTransform, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresHold[0]) & (gray <= thresHold[1])] = 1


canv, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(24, 9))
canv.tight_layout()
ax1.set_title('Original', fontsize=30)
ax1.imshow(testImgToColorTransform)
ax2.set_title('Gray', fontsize=30)
ax2.imshow(gray,cmap='gray')
ax3.set_title('Gray binary', fontsize=30)
ax3.imshow(binary,cmap='gray')

#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.imshow(gray,cmap='gray')
redChannel = testImgToColorTransform[:,:,0]
greenChannel = testImgToColorTransform[:,:,1]
blueChannel = testImgToColorTransform[:,:,2]

canv, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
canv.tight_layout()
ax1.set_title('Red', fontsize=30)
ax1.imshow(redChannel,cmap='gray')
ax2.set_title('Green', fontsize=30)
ax2.imshow(greenChannel,cmap='gray')
ax3.set_title('Blue', fontsize=30)
ax3.imshow(blueChannel,cmap='gray')

`
[color_transforms]: ./examples/color_transforms.png "Color transforms"


#### The Red channel looks great! lets see it in binary
`

thresHoldToRed = (200, 255)
binaryRed = np.zeros_like(redChannel)
binaryRed[(redChannel > thresHoldToRed[0]) & (redChannel <= thresHoldToRed[1])] = 1


canv, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
canv.tight_layout()
ax1.set_title('Red Channel', fontsize=50)
ax1.imshow(redChannel,cmap='gray')
ax2.set_title('Red binary', fontsize=50)
ax2.imshow(binaryRed,cmap='gray')

`
[color_transforms]: ./examples/color_transforms2.png "Color transforms"


#### Lets look at HLS channels
Read more about HLS colorspaces:
[HLS](https://en.wikipedia.org/wiki/HSL_and_HSV)

`

hlsChannels = cv2.cvtColor(testImgToColorTransform, cv2.COLOR_RGB2HLS)
hChannel = hlsChannels[:,:,0]
lChannel = hlsChannels[:,:,1]
sChannel = hlsChannels[:,:,2]

canv, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
canv.tight_layout()
ax1.set_title('H channel', fontsize=30)
ax1.imshow(hChannel,cmap='gray')
ax2.set_title('L channel', fontsize=30)
ax2.imshow(lChannel,cmap='gray')
ax3.set_title('S channel', fontsize=30)
ax3.imshow(sChannel,cmap='gray')

`
[color_transforms]: ./examples/color_transforms3.png "Color transforms"


#### The S channel picks up the lines well, so let's try applying a threshold there:
`

threshToSChannel = (100, 255)
sBinary = np.zeros_like(sChannel)
sBinary[(sChannel > threshToSChannel[0]) & (sChannel <= threshToSChannel[1])] = 1

canv, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
canv.tight_layout()
ax1.set_title('S Channel', fontsize=50)
ax1.imshow(sChannel,cmap='gray')
ax2.set_title('S binary', fontsize=50)
ax2.imshow(sBinary,cmap='gray')

`
[color_transforms]: ./examples/color_transforms4.png "Color transforms"


#### 3. Gradient Treshold, Sobel Operator, Applying Sobel
Read more about edge detection
[Edge detection](https://en.wikipedia.org/wiki/Edge_detection)
#### Tresholded gradient
`

grad_binary = abs_sobel_thresh(testImgToColorTransform, orient='x', thresh_min=10, thresh_max=100)
# Plot the result
canv, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
canv.tight_layout()
ax1.imshow(testImgToColorTransform)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

`
[color_transforms]: ./examples/color_transforms5.png "Color transforms"

#### Magnitude of the gradient

`
mag_binary = tresholdedMagnitude(testImgToColorTransform, sobel_kernel=9, mag_thresh=(30, 100))
# Plot the result
canv, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
canv.tight_layout()
ax1.imshow(testImgToColorTransform)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
`
[color_transforms]: ./examples/color_transforms6.png "Color transforms"

#### Direction of the gradient
`
dir_binary = tresholdedDirection(testImgToColorTransform, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
canv, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
canv.tight_layout()
ax1.imshow(testImgToColorTransform)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
`
[color_transforms]: ./examples/color_transforms7.png "Color transforms"

## 4. Apply a perspective transform to rectify binary image ("birds-eye view").
To this transformation you 'll need source and destination points. To make it easier i made a function for you called "markerDotHelper".
It shows the given array of points on the image that you want.
´
testImage2 = plt.imread('./test_images/test2.jpg')
h,w = testImage2.shape[:2]

src = np.float32([ 
                (550, 450)
                ,(720, 450)
                ,(260, 680)
                ,(1160, 680)
                ])

dst = np.float32([ 
                (430, 0)
                ,(900, 0)
                ,(430, h)
                ,(900, h)
                ])
canv, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(24, 9))
canv.tight_layout()
ax1.set_title('Lane area', fontsize=20)
ax2.set_title('Lane area bird\'s eye view', fontsize=20)
#markerDotHelper(testImgToColorTransform,src)
markedImg=markerLineHelper(src,testImage2,ax1,True,True)
uN, M  = unwarp(testImage2,src,dst)
markerLineHelper(dst,uN,ax2,True,True)

hlsChannels = cv2.cvtColor(testImage2, cv2.COLOR_RGB2HLS)
sCT = hlsChannels[:,:,2]
threshToSChannel = (100, 255)
sTBinary = np.zeros_like(sCT)
sTBinary[(sCT > threshToSChannel[0]) & (sCT <= threshToSChannel[1])] = 1

unwarpedBinaryImage, M = unwarp(sTBinary,src,dst)
ax3.set_title('S channel', fontsize=20)
ax3.imshow(unwarpedBinaryImage,cmap='gray')

ax4.set_title('S channel Binary Bird\'s eye interesting area', fontsize=20)
r = markerLineHelper(dst,unwarpedBinaryImage,ax4,True,True)

´
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
[birdseye]: ./examples/birdseye.png "Bird's eye view"


## 5. Detect lane pixels and fit to find the lane boundary.

`
testImage = plt.imread('./test_images/test3.jpg')
img_size = (testImage.shape[1], testImage.shape[0])
redCha = testImage[:,:,0]
hlsC= cv2.cvtColor(testImage, cv2.COLOR_RGB2HLS)
sChan = hlsC[:,:,2]
lChan = hlsC[:,:,1]

threshToSC = (125, 255)
sBinary = np.zeros_like(sChan)
sBinary[(sChan > threshToSC[0]) & (sChan <= threshToSC[1])] = 1

threshToLC = (180, 255)
lBinary = np.zeros_like(lChan)
lBinary[(lChan > threshToLC[0]) & (lChan <= threshToLC[1])] = 1


threshToRC = (225, 255)
rBinary = np.zeros_like(redCha)
rBinary[(redCha > threshToRC[0]) & (redCha <= threshToRC[1])] = 1


h,w = testImage.shape[:2]

sourcePoints = np.float32([ 
                            (580, 450)
                            ,(680, 450)
                            ,(150, 680)
                            ,(1200, 680)
                            ])

destinationPoints = np.float32([ 
                            (430, 0)
                            ,(900, 0)
                            ,(430, h)
                            ,(900, h)
                            ])
'''
gradx = abs_sobel_thresh(imageParam, orient='x', thresh_min=10, thresh_max=255)
grady = abs_sobel_thresh(imageParam, orient='y',  thresh_min=25, thresh_max=255)
dir_binary = tresholdedDirection(imageParam, sobel_kernel=7, thresh=(0, 0.0777))

uGradx, M = unwarp(gradx,sourcePoints,destinationPoints)
uGrady, M = unwarp(grady,sourcePoints,destinationPoints)

uDir, M = unwarp(dir_binary,sourcePoints,destinationPoints)
'''
gradx = abs_sobel_thresh(testImage, orient='x', thresh_min=20, thresh_max=255)
uGradx, M = unwarp(gradx,sourcePoints,destinationPoints)
uSBinaryy, M = unwarp(sBinary,sourcePoints,destinationPoints)
uLBinaryy, M = unwarp(lBinary,sourcePoints,destinationPoints)
uRBinaryy, M = unwarp(rBinary,sourcePoints,destinationPoints)
cBinary = np.zeros_like(uSBinaryy)*255
#cBinary[( (uSBinaryy==1)) & (uRBinaryy==1) |  (uLBinaryy==1)] = 1
#cBinary[((uBinaryMag==1) | (uSBinaryy==1)) | (uRBinaryy==1) |  (uLBinaryy==1)] = 1

cBinary[( (uRBinaryy==1)) | (uGradx==1)] = 1

canv, (bx1,bx2,bx3) = plt.subplots(1, 3, figsize=(24, 9))
canv.tight_layout()
bx1.set_title('Combined channel Binary Bird\'s eye interesting area', fontsize=20)
bx2.set_title('Detected lane pixels with polynomial fit', fontsize=20)
#markerLineHelper(destinationPoints,cBinary,bx1,True,True)
bx1.imshow(cBinary,cmap='gray')
out_img = fit_polynomial(cBinary)
processor = LaneProcessor()
lanesOnImg = processor.process_image(testImage)
#drawLanesPipeline(testImgToSobel,combinedBinary,vertices)
bx2.imshow(lanesOnImg)

#plt.imshow(resultImg)
`
## 5. Detect lane pixels and fit to find the lane boundary.
Line Finding Method: Peaks in a Histogram
After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly.
However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.
Plotting a histogram of where the binary activations occur across the image is one potential solution for this.

In find_lane_pixels method I use a histogram to find peaks:

`
def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, fitted = find_lane_pixels(binary_warped)
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    fitted[lefty, leftx] = [255, 0, 0]
    fitted[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(fitted)
    return fitted
`

[Detected Lanes]: ./examples/detected_lanes2.png "Detected Lanes"
#### Polynomial fit values
[Detected Lanes]: ./examples/detected_lanes3.png "Detected Lanes"

## 6. Determine the curvature of the lane and vehicle position with respect to center.
`
def measure_curvature_real(binary,leftx, lefty, rightx, righty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    h,w = binary.shape[:2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    #https://www.sciencedirect.com/topics/engineering/road-width
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Start by generating our fake example data
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    distanceFromCenter = 0
    if left_fit is not None and right_fit is not None:
        
        lX = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        rX = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        
        positionOfTheCar = w/2;#center of the img
        centerOfLane = (lX+rX)/2
        distanceFromCenter = (positionOfTheCar-centerOfLane)*xm_per_pix
    
    return distanceFromCenter, left_curverad, right_curverad


#USAGE:
imageP = testImage#plt.imread('./test_images/test1.jpg')
img_size = (imageP.shape[1], imageP.shape[0])

sourcePoints = np.float32(
                [[(img_size[0] / 2) - 90, img_size[1] / 2 + 100],
                [((img_size[0] / 6) - 10), img_size[1]],
                [(img_size[0] * 5 / 6) + 60, img_size[1]],
                [(img_size[0] / 2 + 90), img_size[1] / 2 + 100]])
destinationPoints = np.float32(
                [[(img_size[0] / 4), 0],
                [(img_size[0] / 4), img_size[1]],
                [(img_size[0] * 3 / 4), img_size[1]],
                [(img_size[0] * 3 / 4), 0]])

Canny = cv2.Canny(imageP,100,150)
unwarpedCanny, M = unwarp(Canny,sourcePoints,destinationPoints)

imageParam = imageP#cal_undistort(imageP,objpoints,imgpoints)

thresHoldGray = (200, 255)
gray = cv2.cvtColor(imageP, cv2.COLOR_RGB2GRAY)
gBinary = np.zeros_like(gray)
gBinary[(gray > thresHoldGray[0]) & (gray <= thresHoldGray[1])] = 1

redCha = imageParam[:,:,0]
hlsC= cv2.cvtColor(imageParam, cv2.COLOR_RGB2HLS)
sChan = hlsC[:,:,2]
lChan = hlsC[:,:,1]

threshToSC = (220, 255)
sBinary = np.zeros_like(sChan)
sBinary[(sChan > threshToSC[0]) & (sChan <= threshToSC[1])] = 1

threshToLC = (199, 255)
lBinary = np.zeros_like(lChan)
lBinary[(lChan > threshToLC[0]) & (lChan <= threshToLC[1])] = 1


threshToRC = (200, 255)
rBinary = np.zeros_like(redCha)
rBinary[(redCha > threshToRC[0]) & (redCha <= threshToRC[1])] = 1
h,w = imageParam.shape[:2]    

uSBinaryy, M = unwarp(sBinary,sourcePoints,destinationPoints)
uRBinaryy, M = unwarp(rBinary,sourcePoints,destinationPoints)
uGBinaryy, M = unwarp(gBinary,sourcePoints,destinationPoints)

cBinary = np.zeros_like(uSBinaryy)*255
unCanBin = np.zeros_like(unwarpedCanny)
unCanBin[(unwarpedCanny > 120) & (unwarpedCanny <= 255)]=1

cBinary[(unCanBin==1)&((uRBinaryy==1) | (uSBinaryy==1))  ] = 1
vehiclePos ,left_curverad, right_curverad = measure_curvature_real(cBinary,leftx, lefty, rightx, righty)
print('Left radius of curvature:',left_curverad)
print('Right radius of curvature:',right_curverad)
print('Vehicle Position from center:',vehiclePos)
print('Curve radius:',(left_curverad+right_curverad)/2)
'''
cv2.putText(imageP, 'Radius of curvature = ' + '{:03.0f}'.format((left_curverad+right_curverad)/2) + '(m)', (20,60), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 2, cv2.FILLED)
leftOrRight=''
if vehiclePos<0:
    leftOrRight='left'
else:
    leftOrRight='right'
    
cv2.putText(imageP, 'Vehicle is ' + '{:0.3f}'.format(abs(vehiclePos)) + '(m) '+leftOrRight+' of center', (20,120), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 2, cv2.FILLED)
plt.imshow(imageP)
'''
`
Example result:
Left radius of curvature: 831.7235357109412
Right radius of curvature: 284.00227183320334
Vehicle Position from center: -0.15515864233115875
Curve radius: 557.8629037720723

## 7. Warp the detected lane boundaries back onto the original image.

I made a configurable "LaneProcessor" class to detect lane boundaries, warp, distort,etc.
First step is making a binary image to find easier the lanes on the warped image 
, next step is find lane pixels on it, fit the polyline store the data in Line class.
You can set up some things like changeDataWriteState(). It means you turn on or turn of the data write state if you want.
If the data write state is on, the wanted curvature and center measures are appearing on the image. The default state is off.

`
testIma = plt.imread('./test_images/test2.jpg')
prc=LaneProcessor()
lanesOnImg = prc.process_image(testIma)
plt.imshow(lanesOnImg)
`
[Detected Lanes]: ./examples/back_to_image.png "Detected Lanes"

---

### Pipeline (video)
[Detected Lanes with data]: ./examples/displayed.png "Detected Lanes with data"

Here's a [link to my video result](./videos_output/project_videores.mp4)

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
It works well in a mean time but it works not that easy when lights come on to the picture, the curves are more curved. I can store more data in the LaneProcessor to calculate more. Could be better.

I tried so many filters and mix them but they failed most of the time on the harder video, maybe later i 'll come back with a better filter to make a better solution on every challange video
[Sharper](https://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv) Points
[Masking](https://note.nkmk.me/en/python-opencv-numpy-alpha-blend-mask/) Masking
[Color filters](https://pythonprogramming.net/color-filter-python-opencv-tutorial/) Color filters

Long and short of it is
It was a great experience thanks [Udacity](https://www.udacity.com/) to teaching me these techniques. Let's see what you can create!
Thanks for reading!
