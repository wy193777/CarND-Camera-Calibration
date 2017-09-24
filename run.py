from laneline import calibrate_camera, cal_undistort, compute_perspective_trans_M
from laneline import edge_detect, perspective_trans, find_lines
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import misc
from os import path


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
    M, Minv = compute_perspective_trans_M()
    result = perspective_trans(image, M)

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


def persp_binary_trans(img):
    return edge_detect(perspective_trans(
        cal_undistort(img, objpoints, imgpoints)))[0]


def undistort_persp_trans(img):
    return perspective_trans(cal_undistort(img, objpoints, imgpoints))


def undistort_trans(img):
    return cal_undistort(img, objpoints, imgpoints)


def generate_edge_images():

    def binary_trans(img):
        return edge_detect(img)

    generate_images(
        glob.glob('output_images/test_images_birds_eye/**'),
        'output_images/test_images_binary', binary_trans)


def draw_windows(binary_image_path):
    binary_warped = misc.imread(binary_image_path)
    find_lines(binary_warped)


if __name__ == "__main__":
    # draw_undistord()
    # draw_edges()
    generate_edge_images()
    # draw_perspective()
    # for path in glob.glob('output_images/test_images_birds_eye/*.jpg'):
    #     print(path)
    #     draw_windows(path)
