# importing cv2
import cv2


def setTemplate(image, region):
    template = []
    for i in range(len(region)):
        r = region[i]
        img_crop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        template.append(img_crop)

    return template
