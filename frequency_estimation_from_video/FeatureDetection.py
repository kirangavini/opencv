import numpy as np
import cv2
import os


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


def feature_tracking(new_template, template):
    # Make output dir
    if not os.path.isdir('Translation_Feature'):
        os.mkdir('Translation_Feature')

    path = 'Translation_Feature'
    translation = np.zeros([len(template), 2])
    dst_points = []
    for i in range(len(template)):
        new_t = new_template[i]
        t = template[i]

        # orb = cv2.ORB_create(edgeThreshold=15, patchSize=10, nlevels=8, nfeatures=10, scaleFactor=1.2, scoreType=cv2.ORB_FAST_SCORE)
        orb = cv2.ORB_create(edgeThreshold=1)
        kpt1, des1 = orb.detectAndCompute(t, None)
        kpt2, des2 = orb.detectAndCompute(new_t, None)


        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        tran = [np.mean(dst_pts[:, 0, 0] - src_pts[:, 0, 0]),
                np.mean(dst_pts[:, 0, 1] - src_pts[:, 0, 1])]
        translation[i, :] = np.array(tran)
        new_path = os.path.join(path, str(i + 1) + '_point.txt')
        append_new_line(new_path, ','.join([str(i) for i in tran]))
        dst_points.append(dst_pts)

    return translation, dst_points


if __name__ == '__main__':
    # Read the query image as query_img
    # and traing image This query image
    # is what you need to find in train image
    # Save it in the same directory
    # with the name image.jpg
    query_img = cv2.imread('Humen.jpg')
    train_img = cv2.imread('first_frame.jpg')

    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)

    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatches(query_img, queryKeypoints,
                                train_img, trainKeypoints, matches[:10], None)

    final_img = cv2.resize(final_img, (1000, 650))

    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)
