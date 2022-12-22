import numpy as np
import cv2 as cv
from utils import cv_show, cv_write
from homography import Homography
from matcher import Matcher
from define import PIC_MATCH

class Stitcher:

    # panoramic image stitching
    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False):
        print("==================Stitching begin==================")
        # unpack the images
        imageB, imageA = images

        # find the keypoints and descriptors with SIFT
        kpA, desA = self.sift(imageA)
        kpB, desB = self.sift(imageB)

        # match features between the two images
        matcher = Matcher()
        matches = matcher.knnMatch(desA, desB, ratio)

        # if the matches are less than 4, then there are not enough matches to create a panorama
        print('matches {} keypoints'.format(len(matches)))
        if len(matches) <= 4:
            print("Not enough matches are found")
            return None

        # compute the homography between the two sets of matched points
        ptsA = np.array([kpA[i] for (_, i) in matches]).astype(np.float32)
        ptsB = np.array([kpB[i] for (i, _) in matches]).astype(np.float32)

        (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)
        print("opencv H: {}".format(H))

        homography = Homography()
        (H, status) = homography.findHomography(ptsA, ptsB, cv.RANSAC, 5.0) # H is 3x3 homography matrix   
        print("our H: {}".format(H))

        # show the matches
        matchImage = self.drawMatches(imageA, imageB, kpA, kpB, matches, status)
        if showMatches:
            cv_show("Keypoint Matches", matchImage)
            cv_write(PIC_MATCH, matchImage)
         
        # apply a perspective transform to stitch the images together
        result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0] + imageB.shape[0]))  # TODO: decide the shape
        cv_show('result A', result)
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        cv_show('result A+B', result)

        print("==================Stitching end==================")
        return (result, matchImage)

    def sift(self, image):
        # from opencv documentation
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        descriptor = cv.SIFT_create()
        kp, des = descriptor.detectAndCompute(gray, None)
        kp = np.array([p.pt for p in kp]).astype(np.float32)
        return kp, des

    def drawMatches(self, imageA, imageB, kpA, kpB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3)).astype(np.uint8)
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches 
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis