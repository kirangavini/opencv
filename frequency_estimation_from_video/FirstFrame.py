import cv2


def getFirstFrame(videofile):
    """get the first frame from video"""
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file
        return image
    else:
        print('There is no frame!')
