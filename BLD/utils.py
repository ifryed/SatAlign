from cv2 import ORB_create, BFMatcher, NORM_HAMMING, RANSAC, findHomography, warpPerspective
import numpy as np

def align(img, base_img):
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = ORB_create(500)
    (kpsA, descsA) = orb.detectAndCompute((base_img).astype(np.uint8), None)
    (kpsB, descsB) = orb.detectAndCompute((img).astype(np.uint8), None)

    # match the features
    matcher = BFMatcher(NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descsA, descsB)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * 0.40)
    matches = matches[:keep]

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = findHomography(ptsB, ptsA, method=RANSAC)  # ,maxIters=100)
    # (H, mask) = cv2.estimateAffine2D(ptsB, ptsA, )  # ,maxIters=100)
    # use the homography matrix to align the images
    (h, w) = base_img.shape[:2]
    aligned = warpPerspective(img, H, (w, h))
    # aligned = cv2.warpAffine(img, H, (w, h))

    return aligned