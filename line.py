import cv2
import laneline
import glob
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import pickle
from pathlib import Path


# Define a class to receive the characteristics of each line detection
class Line():

    MATRIX_PATH = 'matrix.p'
    X_M_PER_PIX = 3.7 / 700

    def __init__(self, objpoints, imgpoints):
        self.objpoints = objpoints
        self.imgpoints = imgpoints
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        matrix = Path(self.MATRIX_PATH)
        if matrix.is_file():
            two_M = pickle.load(open(self.MATRIX_PATH, "rb"))
            self.M = two_M['M']
            self.Minv = two_M['Minv']
        else:
            M, Minv = laneline.compute_perspective_trans_M()
            self.M = M
            self.Minv = Minv
            pickle.dump({'M': M, 'Minv': Minv}, open(self.MATRIX_PATH, 'wb'))

    def draw_text(self, frame, text, position, thickness=3, size=2,):

        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), thickness)

    def process(self, img):
        image = img.copy()
        camera_position = img.shape[1] / 2
        calibrated_image = laneline.cal_undistort(
            image, self.objpoints, self.imgpoints
        )
        import ipdb; ipdb.set_trace()
        bird_eye_img = laneline.perspective_trans(calibrated_image, self.M)
        edge_img = laneline.edge_detect(bird_eye_img)
        bird_lane_mask, radius, offset = laneline.find_lines(edge_img)
        persp_lane_mask = laneline.perspective_trans(bird_lane_mask, self.Minv)
        image[:, :, 1] = image[:, :, 1] + persp_lane_mask[:, :, 1]
        image = image.astype(np.uint8)
        self.draw_text(image, "radius: {0:.2f}".format(radius), (50, 100))
        self.draw_text(image, "offset: {0:.2f}".format(offset), (50, 200))
        return image


def process_video(video_path, output_path):
    objpoints, imgpoints = laneline.calibrate_camera(
        glob.glob('camera_cal/*.jpg'))
    processor = Line(objpoints, imgpoints)
    clip1 = VideoFileClip(video_path)
    output_video = clip1.fl_image(processor.process)
    output_video.write_videofile(output_path, audio=False)


def process_image(image_path):
    objpoints, imgpoints = laneline.calibrate_camera(
        glob.glob('camera_cal/*.jpg'))
    processor = Line(objpoints, imgpoints)
    image = misc.imread(image_path)
    return processor.process(image)


if __name__ == "__main__":
    plt.imshow(process_image('test_images/test7_white_road.png'))
    plt.show()
    # process_video(
    #     "project_video.mp4",
    #     "output_videos/project_video.mp4")
    # process_video(
    #     "challenge_video.mp4",
    #     "output_videos/challenge_video.mp4")
    # process_video(
    #     "harder_challenge_video.mp4",
    #     "output_video/harder_challenge_video.mp4")
