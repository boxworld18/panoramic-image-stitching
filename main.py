import cv2 as cv
import numpy as np
import sys, os, math, argparse
from matplotlib import pyplot as plt

PIC_1 = 'pics/snow_left.jpg'
PIC_2 = 'pics/snow_right.jpg'
PIC_SIFT = 'pics/snow_sift.jpg'
PIC_OUT = 'pics/snow_out.jpg'

READ_FLAG = cv.IMREAD_COLOR

def cv_read(img_path, flag=READ_FLAG):
    img = cv.imread(img_path, flag)
    print('reading image: {}, shape: {}'.format(img_path, img.shape))
    return img

def cv_write(img_path, img):
    print('writing image: {}, shape: {}'.format(img_path, img.shape))
    cv.imwrite(img_path, img)    

def cv_show(name, img):
    print('showing image: {}, shape: {}'.format(name, img.shape))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def sift(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()

    # matches = bf.knnMatch(des1, des2, k=2)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)
    # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv_write(PIC_SIFT, img3)

    return img3

def main():
    # read an image
    img1 = cv_read(PIC_1)
    img2 = cv_read(PIC_2)

    # sift
    img3 = sift(img1, img2)

if __name__ == '__main__':
    main()