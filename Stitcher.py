import numpy as np
import cv2 as cv
import math
from utils import cv_show, cv_write
from homography import Homography
from matcher import Matcher
from fusion import Fusion
from define import *

class Stitcher:

    def __init__(self):
        pass

    # panoramic image stitching
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, fusionMethod="default"):
        print("==================Stitching begin==================")
        n = len(images)
        
        # A B C       0 -> 1 into 1 | -1 -> -2 into -2 |                                                    final: 3 // 2 = 1
        # A B C D     0 -> 1 into 1 | -1 -> -2 into -2 | 1 -> 2 into 2 |                                    final: 4 // 2 = 2   
        # A B C D E   0 -> 1 into 1 | -1 -> -2 into -2 | 1 -> 2 into 2 | -2 -> -3 into -3 |                 final: 5 // 2 = 2
        # A B C D E F 0 -> 1 into 1 | -1 -> -2 into -2 | 1 -> 2 into 2 | -2 -> -3 into -3 | 2 -> 3 into 3 | final: 6 // 2 = 3

        i = cnt = 0
        reverse = True
        imageA = imageB = None
        while cnt < n - 1:
            if reverse: # imageA -> imageB
                imageA = images[i]
                imageB = images[i + 1]
                reverse = False
                print("imageA: {} -> imageB: {}".format(i, i + 1))
            else: # imageB -> imageA
                imageA = images[-i - 2]
                imageB = images[-i - 1]
                reverse = True
                print("imageB: {} -> imageA: {}".format(-i - 1, -i - 2))

            # reverse == True: imageB -> imageA (left + H[right])
            # reverse == False: imageA -> imageB (H[left] + right)

            # cylindrical projection (useless feature)
            # imageA = self.cylindricalProjection(imageA)
            # imageB = self.cylindricalProjection(imageB)
            
            # find the keypoints and descriptors with SIFT
            kpA, desA = self.sift(imageA)
            kpB, desB = self.sift(imageB)

            # match features between the two images
            matcher = Matcher()
            if reverse: 
                matches = matcher.knnMatch(desB, desA, ratio)
            else: 
                matches = matcher.knnMatch(desA, desB, ratio)

            # # compute the homography between the two sets of matched points
            if reverse:
                (H, status) = self.findHomography(kpB, kpA, matches, reprojThresh)
            else:
                (H, status) = self.findHomography(kpA, kpB, matches, reprojThresh)

            # show the matches
            # matchImage = self.drawMatches(imageA, imageB, kpA, kpB, matches, status)    
            
            # apply a perspective transform to stitch the images together
            result = self.fusion((imageA, imageB), H, fusionMethod, reverse=reverse)
            if reverse:
                images[-i - 2] = result
                i += 1
            else:
                images[i + 1] = result
            cnt += 1

        print("==================Stitching end==================")
        return images[n // 2]

    # do cylindrical projection
    def cylindricalProjection(self, image, f = 1.0):
        h = image.shape[0]
        w = image.shape[1]
        
        res = np.zeros_like(image)
        # f = (w * 0.5) / math.atan2(np.pi, 8.0)
        f = 500.0
        
        for i in range(h):
            for j in range(w):
                x = int(f * math.atan2(j - w * 0.5, f) + f * math.atan2(w, 2.0 * f) + 0.5)
                y = int(f * (i - h * 0.5) * 1.0 / np.sqrt((j - w * 0.5) * (j - w * 0.5) + f * f) + h * 0.5 + 0.5)
                if x >= 0 and x < w and y >= 0 and y < h:
                    res[y, x] = image[i, j]
        
        return res
    
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

        # homography = Homography()
        # (H, status) = homography.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh) # H is 3x3 homography matrix   
        # print("our H: {}".format(H))

        return (H, status)

    def fusion(self, images, H, fusionMethod, reverse=False):
        imageA, imageB = images
        h1, w1 = imageA.shape[0], imageA.shape[1]
        h2, w2 = imageB.shape[0], imageB.shape[1]
        m1 = np.array([[[0, 0]],[[0, h1]],[[w1, h1]],[[w1, 0]]]).astype(np.float32)
        m2 = np.array([[[0, 0]],[[0, h2]],[[w2, h2]],[[w2, 0]]]).astype(np.float32)
        m2 = cv.perspectiveTransform(m2, H)
        m = np.concatenate((m1, m2), axis=0)
        xmin, ymin = (np.min(m, axis=0).ravel() - 0.5).astype(np.int32)
        xmax, ymax = (np.max(m, axis=0).ravel() + 0.5).astype(np.int32)
        t = [-xmin, -ymin]
        H_pers = np.array([[1, 0, t[0]],[0, 1, t[1]],[0, 0, 1]]).dot(H) # translate
        h, w = (ymax - ymin, xmax - xmin)

        print('h1: {}, h2: {}, w1: {}, w2: {}'.format(h1, h2, w1, w2))
        print('h: {}, w: {}'.format(h, w))
        print('t: {}'.format(t))
        
        result = None
        if reverse: 
            result = cv.warpPerspective(imageB, H_pers, (w, h)) # 变换右侧图像
        else:
            result = cv.warpPerspective(imageA, H_pers, (w, h)) # 变换左侧图像

        img = imageA if reverse else imageB
        h_top = min(img.shape[0]+t[1], h)
        w_top = min(img.shape[1]+t[0], w)
        fusion = Fusion()
        if fusionMethod == "poisson":
            result = fusion.poisson(result, img, reverse, t, h_top, w_top)
        elif fusionMethod == "weight": 
            result = fusion.weigh_fussion(result, img, reverse, t, h_top, w_top)
        elif fusionMethod == "multiband":
            result = fusion.Multiband(result, img, reverse, t, h_top, w_top)
        else:
            result = fusion.simple(result, img, reverse, t, h_top, w_top)
        print('result shape: {}'.format(result.shape))

        BOUND = 4000
        result = result[t[1]:h_top,:,:]
        result = result[:,:BOUND,:] if reverse else result[:,-BOUND:,:]
        result = self.cut(result)

        return result

    def cut(self, result):
        cnt = cut1 = cut2 = 0
        flag = False
        for j in range(0, result.shape[1]): # 从左到右
            cnt = 0
            for i in range(0, result.shape[0]): # 从上到下
                if result[i][j][0] == 0 or result[i][j][1] == 0 or result[i][j][2] == 0:
                    cnt += 1
            if flag == False and cnt < result.shape[0] * 0.8:
                cut1 = j
                flag = True
            elif flag == True and cnt >= result.shape[0] * 0.8:
                cut2 = j
                break
        if cut2 == 0:
            cut2 = result.shape[1]
        result = result[:, cut1:cut2, :]

        cnt = cut1 = cut2 = 0
        flag = False
        for i in range(0, result.shape[0]): # 从左到右
            cnt = 0
            for j in range(0, result.shape[1]): # 从上到下
                if result[i][j][0] == 0 or result[i][j][1] == 0 or result[i][j][2] == 0:
                    cnt += 1
            if flag == False and cnt < result.shape[1] * 0.3:
                cut1 = i
                flag = True
            elif flag == True and cnt >= result.shape[1] * 0.3:
                cut2 = i
                break
        if cut2 == 0:
            cut2 = result.shape[0]
        result = result[cut1:cut2:, :, :]
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