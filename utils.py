import cv2 as cv

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