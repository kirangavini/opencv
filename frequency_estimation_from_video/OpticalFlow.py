import cv2 as cv
import numpy as np
import os
from SelectROI import ROISelection
from FirstFrame import getFirstFrame
from DrawRectangle import drawROI
from Template import setTemplate
from ImageRegistration import subpixel_translation
from animate import animate


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def estimate_FLK(new_template, pre_template, prev_featurepoint):
    # Collect current feature points
    curr_featurepoint = []
    # Collect current templates
    curr_template = []
    feature_params = dict(maxCorners=300, qualityLevel=0.05, minDistance=20, blockSize=10)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Make output dir
    if not os.path.isdir('Translation_OpticalFlow'):
        os.mkdir('Translation_OpticalFlow')

    path = 'Translation_OpticalFlow'
    old_featurepoints = []
    new_featurepoints = []
    trans = np.zeros([len(pre_template), 2])
    for i in range(len(pre_template)):
        x = len(pre_template)
        new_t = new_template[i]
        t = pre_template[i]
        if len(prev_featurepoint) == 0:
            # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
            # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
            prev = cv.goodFeaturesToTrack(t, mask=None, **feature_params)
        else:
            prev = prev_featurepoint[i]

        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        mask = np.zeros_like(prev)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(t, new_t, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        old_featurepoints.append(good_old)
        # Selects good feature points for next position
        good_new = next[status == 1]
        new_featurepoints.append(good_new)
        tran = [np.mean(good_new.reshape(-1, 1, 2)[:, 0, 0] - good_old.reshape(-1, 1, 2)[:, 0, 0]),
                np.mean(good_new.reshape(-1, 1, 2)[:, 0, 1] - good_old.reshape(-1, 1, 2)[:, 0, 1])]
        pre_t = new_t.copy()
        curr_featurepoint.append(good_new.reshape(-1, 1, 2))
        curr_template.append(pre_t)
        trans[i, :] = np.array(tran)
        new_path = os.path.join(path, str(i + 1) + '_point.txt')
        append_new_line(new_path, ','.join([str(i) for i in tran]))

    return curr_template, curr_featurepoint, trans, old_featurepoints, new_featurepoints


"""
shift, error, diffphase = register_translation(new_t, t, 10)
translation[i, :] = np.array(shift)
new_path = os.path.join(path, str(i + 1) + '_point.txt')
append_new_line(new_path, ','.join([str(i) for i in shift]))
# translation.append(shift)
# translation = np.array(translation)
# print(translation)
return translation
"""
