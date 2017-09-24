

[//]: # (Image References)
[calibrated5]: ./output_images/camera_cal/calibration5.jpg
[undistorted_test_image]: ./output_images/test_images_undistored/test7_white_road.png
[white_road]: ./test_images/test7_white_road.png
[birds_eye]: ./output_images/test_images_birds_eye/straight_lines1.jpg
[p_fit]: ./output_images/test_image_rect/test5.png
[output_image]: ./output_images/frame_result.png
### Camera Calibration

#### 1. Calibrate Camera.

The code for this step is contained in `./laneline.py` through line 9 to 39.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Calibrated Chessboard][calibrated5]

### Pipeline (single images)

#### 1. Undistort Images:

Steps for this is as same as calibrate chessboard images.

An example output is:


![alt text][undistorted_test_image]

#### 2. Perspective Transform

The code for my perspective transform includes a function called `perspective_trans()`, which appears in lines 239 through 244 in the file `laneline.py`.  The `perspective_trans()` function takes as inputs an image (`img`), as well as the matrix to transform the image.

There is another method that called `compute_perspective_trans_M()` contains the hardcode source and destination points in the following manner:

```python
w, h = 1280, 720
x, y = 0.5*w, 0.8*h
src = np.float32([
    [200./1280*w, 720./720*h],
    [453./1280*w, 547./720*h],
    [835./1280*w, 547./720*h],
    [1100./1280*w, 720./720*h]])
dst = np.float32([
    [(w-x)/2., h],
    [(w-x)/2., 0.82*h],
    [(w+x)/2., 0.82*h],
    [(w+x)/2., h]])
```

This resulted in the following source and destination points:

| Source        | Destination       |
|:-------------:|:-----------------:|
| 200, 720      | 320, 720          |
| 453, 547      | 320, 590.40002441 |
| 835, 547      | 960, 590.4000244  |
| 1100, 720     | 960, 720          |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image like below:

![Birds Eye View][birds_eye]

#### 3. Create Threshold Binary Image

I used color to generate a binary image (thresholding steps at lines 84 through 117 in `./laneline.py`). I tried different combination's of RGB, HSV and HLS with gradient thresholding methods. But all my solution has some difficulties on find yellow line on "white" road like bellow:

![White Background Road][white_road]

#### 4. Identified Lane-Line Pixels and Fit a Polynomial

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][p_fit]

The code is start from line 120 in `laneline.py`. As the image shows, I used histogram to identify the start position of sliding windows, then add windows from the bottom to the top.
After that, I used pixels inside the windows to fit the 2nd order polynomial line.  

#### 5. Calculate Radius and Car Position

I did this in lines 212 through 218 in my code in `laneline.py` I used the radius of curvature function to calculate radius of the road.

For car position, I calculated the position of middle pixel of the lane, than transform the pixel offset to real world position.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `line.py` in the function `process()`.  Here is an example of my result on a test image:

![alt text][output_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Hard Part of the Project

Actually extract lane line is pretty hard considering the road could have different color with shadows. I read lots of Udacity forum posts and other articles about this. It is more like try without directions for me. My implementation only works on `project_video.mp4`.

#### 2. Potential Improvement

Use informations from a few most recent frames to let the result more robust and improve the performance.
