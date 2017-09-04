import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import misc
from os import path


def calibrate_camera(fnames):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    nx = 9
    ny = 6
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = []
    for fname in fnames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    return objpoints, imgpoints


def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    print(img.shape)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)


def edge_detect(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It
    # might be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


def perspective_trans(img):
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
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped


def draw_undistord():
    objpoints, imgpoints = calibrate_camera(glob.glob('camera_cal/*.jpg'))
    image = misc.imread('camera_cal/calibration1.jpg')
    undistorted = cal_undistort(image, objpoints, imgpoints)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def draw_edges():
    image = misc.imread('test_images/straight_lines2.jpg')
    result = edge_detect(image)

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result[0])
    ax2.set_title('Colorted Result', fontsize=40)

    ax3.imshow(result[1])
    ax3.set_title('Binary Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def draw_perspective():
    image = misc.imread('test_images/straight_lines1.jpg')
    result = perspective_trans(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)
    ax2.set_title('Colorted Result', fontsize=40)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def generate_images(in_fnames, out_directory, func):
    for fname in in_fnames:
        image = misc.imread(fname)
        save_path = path.join(out_directory, path.basename(fname))
        misc.imsave(save_path, func(image))

if __name__ == "__main__":
    # draw_undistord()
    # draw_edges()
    # draw_perspective()
    generate_images(
        glob.glob('test_images/*.jpg'), 'output_images/test_images', perspective_trans)
