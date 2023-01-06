import numpy as np
import cv2 as cv
from utils import cv_show, cv_write
from homography import Homography
from matcher import Matcher
from fusion import Fusion
from define import *

class Stitcher:

    # panoramic image stitching
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, fusionMethod="default", showAny=True):
        print("==================Stitching begin==================")
        n = len(images)

        # find the keypoints and descriptors with SIFT
        reverse = False
        result = images[0]
        for i in range(1, n):
            reverse = not reverse
            imageA = result
            imageB = images[i]
            kpA, desA = self.sift(imageA)
            kpB, desB = self.sift(imageB)

            # match features between the two images
            matcher = Matcher()
            if reverse:
                matches = matcher.knnMatch(desB, desA, ratio)
            else:
                matches = matcher.knnMatch(desA, desB, ratio)

            # compute the homography between the two sets of matched points
            if reverse:
                (H, status) = self.findHomography(kpB, kpA, matches, reprojThresh)
            else:
                (H, status) = self.findHomography(kpA, kpB, matches, reprojThresh)

            # show the matches
            # matchImage = self.drawMatches(imageA, imageB, kpA, kpB, matches, status)    
            
            # apply a perspective transform to stitch the images together
            result = self.fusion((imageA, imageB), H, fusionMethod, reverse=reverse, showAny=showAny)

        print("==================Stitching end==================")
        return result

    def sift(self, image):
        # from opencv documentation
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        descriptor = cv.SIFT_create()
        kp, des = descriptor.detectAndCompute(gray, None)
        kp = np.array([p.pt for p in kp]).astype(np.float32)
        return kp, des

    def findHomography(self, kpA, kpB, matches, reprojThresh):
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
        (H, status) = homography.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh) # H is 3x3 homography matrix   
        print("our H: {}".format(H))

        return (H, status)

    def fusion(self, images, H, fusionMethod, reverse=False, showAny=True):
        imageA, imageB = images

        def warpTwoImages(img1, img2, H, fusionMethod, reverse=False):
            '''warp img2 to img1 with homograph H'''
            h1,w1 = img1.shape[:2]
            h2,w2 = img2.shape[:2]
            pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
            pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
            pts2_ = cv.perspectiveTransform(pts2, H)
            pts = np.concatenate((pts1, pts2_), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin,-ymin]
            Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

            result = None
            if reverse: 
                result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin)) # 变换右侧图像
            else:
                result = cv.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin)) # 变换左侧图像
            cv_show('result A', result)

            img = img2
            fusion = Fusion()
            if fusionMethod == "poisson":
                result = fusion.poisson(result, img)
            elif fusionMethod == "weight": 
                result = fusion.weigh_fussion(result, img)
            elif fusionMethod == "multiband":
                result = fusion.Multiband(result, img)
            else:
                if reverse:
                    for i in range(t[1], h1+t[1]):
                        for j in range(t[0], w1+t[0]):
                            if result[i][j][0] == 0 or result[i][j][1] == 0 or result[i][j][2] == 0:
                                result[i][j] = img1[i-t[1]][j-t[0]]
                    # result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
                else:
                    result[t[1]:h2+t[1],t[0]:w2+t[0]] = img
            return result

        # apply a perspective transform to stitch the images together
        result = cv.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0] + imageB.shape[0]))
        # result = warpTwoImages(imageA, imageB, H, fusionMethod, reverse=reverse)

        if showAny:
            cv_show('result A', result)

        img = imageA
        fusion = Fusion()
        if fusionMethod == "poisson":
            result = fusion.poisson(result, img)
        elif fusionMethod == "weight": 
            result = fusion.weigh_fussion(result, img)
        elif fusionMethod == "multiband":
            result = fusion.Multiband(result, img)
        else:
            result[0:img.shape[0], 0:img.shape[1]] = img
        return result


    def drawMatches(self, imageA, imageB, kpA, kpB, matches, status):
        # initialize the output visualization image
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        matchImage = np.zeros((max(hA, hB), wA + wB, 3)).astype(np.uint8)
        matchImage[0:hA, 0:wA] = imageA
        matchImage[0:hB, wA:] = imageB

        # loop over the matches 
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv.line(matchImage, ptA, ptB, (0, 255, 0), 1)

        cv_write(PIC_MATCH, matchImage)

        return matchImage