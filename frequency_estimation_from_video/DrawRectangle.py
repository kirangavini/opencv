# importing cv2
import cv2


def drawROI(frame, region=None):
    if region == None:
        print("Please Set the ROI!")
    else:
        for i in range(len(region)):
            r = region[i]
            # Start coordinate
            # represents the top left corner of rectangle
            start_point = (int(r[0]), int(r[1]))

            # Ending coordinate
            # represents the bottom right corner of rectangle
            end_point = (int(r[0] + r[2]), int(r[1] + r[3]))

            # color in BGR
            color = (0, 255, 0)

            # Line thickness of 2 px
            thickness = 2

            image = cv2.rectangle(frame, start_point, end_point, color, thickness)

    return image
