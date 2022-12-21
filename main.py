import cv2 as cv
import numpy as np
import sys, os, math, argparse
from matplotlib import pyplot as plt
from Stitcher import Stitcher
from utils import cv_read, cv_show, cv_write

PIC_TAG = 'snow'
PIC_SUFFIX = '.jpg'
PIC_1 = 'pics/{}/{}_left{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
PIC_2 = 'pics/{}/{}_right{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
PIC_SIFT = 'pics/{}/{}_sift{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
PIC_OUT = 'pics/{}/{}_out{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
PIC_RESULT = 'pics/{}/{}_result{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
PIC_CV_RESULT = 'pics/{}/{}_cv_result{}'.format(PIC_TAG, PIC_TAG, PIC_SUFFIX)
    
def main():
    # 读取拼接图片
    imageA = cv_read(PIC_1)
    imageB = cv_read(PIC_2)

    # 把图片拼接成全景图
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    # 显示所有图片
    cv_show("Keypoint Matches", vis)
    cv_write(PIC_SIFT, vis)
    cv_show("Result", result)
    cv_write(PIC_OUT, result)

    # 用opencv自带的函数拼接图片
    stitcher = cv.Stitcher_create()
    (status, pano) = stitcher.stitch([imageA, imageB])
    print('OpenCV Stitcher Status: {}'.format(status))
    if status == cv.Stitcher_OK:
        cv_show("OpenCV Stitcher", pano)
        cv_write(PIC_CV_RESULT, pano)
    
if __name__ == '__main__':
    main()