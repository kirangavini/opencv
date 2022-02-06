"""Import"""
# Import Libs
import numpy as np
import cv2
from imutils.video import FPS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import style
import glob
import os
import io
# Import Functions
from SelectROI import ROISelection
from FirstFrame import getFirstFrame
from DrawRectangle import drawROI
from Template import setTemplate
from ImageRegistration import subpixel_translation
from animate import animate
from OpticalFlow import estimate_FLK
from FeatureDetection import feature_tracking
# plot
import time
import matplotlib.pyplot as plt
from PIL import Image


def drawopticalflow(old_fpts, new_fpts, f, m, r):
    color = (0, 0, 255)
    for k in range(len(r)):
        good_new = new_fpts[k]
        good_old = old_fpts[k]
        rect = r[k]
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            m = cv2.line(m, (a + np.float32(rect[0]), b + np.float32(rect[1])),
                         (c + np.float32(rect[0]), d + np.float32(rect[1])), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            f = cv2.circle(f, (a + np.float32(rect[0]), b + np.float32(rect[1])), 3, color, -1)
    output = cv2.add(f, m)
    return output


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    video_file = 'slomo_1634657955.mov'
    #video_file = 'Slomo_video2.mp4'
    #video_file = 'Tuning Fork in Slow Motion_2.mp4'
    cap = cv2.VideoCapture(video_file)

    # extract the first frame for ROI selection
    first_frame = getFirstFrame(video_file)
    mask = np.zeros_like(first_frame) # make a array of zeros of the size of first frame
    window_name = 'ROI-Selection' # This is just a string variable
    print("Please Select The ROI!") 
    ROI = ROISelection(window_name, first_frame)  # Set the ROI manually
    r = ROI.SetROI(1)  # The region of ROI, set the number of ROI in frame
    if not os.path.isdir('ROI'):
        os.mkdir('ROI')
    roidir = 'ROI/roi.txt'
    np.savetxt(roidir, r, fmt='%.2f')

    cv2.destroyAllWindows()

    # Crop image for template
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    template = setTemplate(first_frame, r)
    translation_x = []
    translation_y = []
    algorithm = 'FeatureDetection'

    
    if algorithm == 'NRMSE':  # Template Matching
        # Remove all text files
        if not os.path.isdir('Translation_NRMSE'):
            os.mkdir('Translation_NRMSE')
        mydir = 'Translation_NRMSE/'
        filelist = glob.glob(os.path.join(mydir, "*"))
        for f in filelist:
            os.remove(f)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(mydir, "output.avi"), fourcc, 30.0, (1280, 720))
        out_fig = cv2.VideoWriter(os.path.join(mydir, "trans_fig.avi"), fourcc, 30.0, (1000, 300), isColor=True)
    elif algorithm == 'OpticalFlow':  # Optical flow
        if not os.path.isdir('Translation_OpticalFlow'):
            os.mkdir('Translation_OpticalFlow')
        mydir = 'Translation_OpticalFlow/'
        filelist = glob.glob(os.path.join(mydir, "*"))
        for f in filelist:
            os.remove(f)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(mydir, "output.avi"), fourcc, 30.0, (1920, 1080))
        out_fig = cv2.VideoWriter(os.path.join(mydir, "trans_fig.avi"), fourcc, 30.0, (1000, 300), isColor=True)
    elif algorithm == 'FeatureDetection':  # Keypoint Tracking
        if not os.path.isdir('Translation_Feature'):
            os.mkdir('Translation_Feature')
        mydir = 'Translation_Feature/'
        filelist = glob.glob(os.path.join(mydir, "*"))
        for f in filelist:
            os.remove(f)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(mydir, "output.avi"), fourcc, 30.0, (1920, 1080))
        out_fig = cv2.VideoWriter(os.path.join(mydir, "trans_fig.avi"), fourcc, 30.0, (1000, 300), isColor=True)

    # Plot Real-Time Translation
    style.use('fivethirtyeight')
    plt.rcParams['animation.html'] = 'jshtml'
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    #fig.show()
    i = 0
    disp_x = []
    disp_y = []
    fig.show()

    # Parameters in optical flow
    prev_featurepoint = []

    # write video


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not ret:
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw ROI on the frame
        frame = drawROI(frame, r)

        # Template in new frame
        new_template = setTemplate(gray, r)

        # Calculate the shift
        # --------------------------------------------------------
        # Choose algorithm
        # algorithm = 'NRMSE'
        # --------------------------------------------------------
        if algorithm == 'NRMSE':
            # algorithm == 'NRMSE'
            shift = subpixel_translation(new_template, template)
            print(shift)
            translation_x.append(shift[0][0])
            translation_y.append(shift[0][1])
        # --------------------------------------------------------
        elif algorithm == 'OpticalFlow':
            # algorithm == 'OpticalFlow':
            curr_template, curr_featurepoint, trans, old_featurepoints, new_featurepoints = estimate_FLK(new_template,
                                                                                                         template,
                                                                                                         prev_featurepoint)
            frame = drawopticalflow(old_featurepoints, new_featurepoints, frame, mask, r)
            template = curr_template
            prev_featurepoint = curr_featurepoint
            if i == 0:
                translation_x.append(trans[0][0])
                translation_y.append(trans[0][1])
            else:
                translation_x.append(translation_x[i - 1] + trans[0][0])
                translation_y.append(translation_y[i - 1] + trans[0][1])
        # --------------------------------------------------------
        elif algorithm == 'FeatureDetection':
            trans, dst_points = feature_tracking(new_template, template)
            print(trans)
            translation_x.append(trans[0][0])
            translation_y.append(trans[0][1])
        # --------------------------------------------------------

        # plot real-time displacement
        disp_x.append(np.float64(i))
        ax.clear()
        line1 = ax.plot(disp_x, translation_y, 'm-', label='x', linewidth=2.0)
        line2 = ax.plot(disp_x, translation_x, 'b-', label='y', linewidth=2.0)
        # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        ax.legend(loc='upper right')
        ax.set_xlim(left=0, right=1.2 * len(disp_x))
        ax.set_ylim(bottom=min(min(translation_x), min(translation_y)) - 1,
                    top=max(max(translation_x), max(translation_y)) + 1)

        # ax.set(xlabel="Frames", ylabel="Displacement (Pixel)")
        ax.grid(True)
        plt.tick_params(labelsize=10)
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 12,
                 }
        plt.xlabel('Frames', font2)
        plt.ylabel('Displacement (Pixel)', font2)
        fig.subplots_adjust(bottom=0.15)
        fig.canvas.draw()
        time.sleep(0.0)
        i += 1

        # print("Detected subpixel offset (y, x): {}".format(shift))

        # Display the resulting frame
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg")
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        fig_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_BGR2RGB)

        # show results
        # cv2.imshow("Displacement", fig_img)
        frame1 = cv2.resize(frame, (512, 512))
        cv2.imshow("Video", frame1)

        # write video
        out_fig.write(fig_img)
        out.write(frame)  # save as video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # draw_translation = np.array(translation)
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
