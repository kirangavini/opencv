import cv2
import numpy as np

""""""
class ROISelection:
    """Select a ROI to be tracked"""
    def __init__(self, windowname, img):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(windowname, img)
        self.showCrosshair = True
        self.fromCenter = False
        self.point = []

    def SetROI(self, count=1, img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.img)
        self.point = []
        for c in range(count):
            region = cv2.selectROI(self.windowname, self.img, self.showCrosshair, self.fromCenter)
            self.point.append(region)
        """
        while 1:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        """
        # cv2.destroyAllWindows()
        return self.point


if __name__ == '__main__':
    # Read image
    im = cv2.imread("Humen.jpg")

    """
    # Select ROI
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", im, fromCenter, showCrosshair)
    """
    windowname = 'ROI'
    ROI = ROISelection(windowname, im)
    r = ROI.SetROI(1)
    r = r[0]
    print(r)
    # Crop image
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
