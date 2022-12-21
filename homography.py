import cv2 as cv
import numpy as np
import random

class Homography:
    def __init__(self):
        pass
    
    def checkCoLinear(self, pts):
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        return x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) == 0
    
    def doRansac(self, src_pts, dst_pts, sigma):
        # parameters init
        pre_total = 0
        max_iters = 1000
        min_iters = 500
        iters = 2000
        count = src_pts.shape[0]
        P = 0.99
        it = 0
        
        # result init
        res_H = np.zeros((3, 3))
        res_mask = np.zeros(count)
        
        while it < min(max_iters, max(iters, min_iters)):
            # select 4 points randomly
            target = random.sample(range(0, count), 4)
            src_pt4 = np.array([src_pts[pt] for pt in target])
            dst_pt4 = np.array([dst_pts[pt] for pt in target])
            
            # check if collinear
            flag1 = self.checkCoLinear(src_pt4[1:]) and self.checkCoLinear(src_pt4[:3])
            flag2 = self.checkCoLinear(dst_pt4[1:]) and self.checkCoLinear(dst_pt4[:3])
            if (flag1 or flag2):
                continue
            
            # calculate homography matrix
            mat_H = self.doKernel(src_pt4, dst_pt4)
            if mat_H is None:
                continue
            
            # count inliers
            cnt = 0
            tmp_mask = np.zeros(count)
            for idx in range(0, count):
                # known points
                org_x, org_y = src_pts[idx]
                tar_x, tar_y = dst_pts[idx]
                
                # predict points
                pre_x, pre_y, scale = np.matmul(mat_H, [org_x, org_y, 1])
                scale = 1e-15 if abs(scale) < 1e-15 else scale
                pre_x, pre_y = pre_x / scale, pre_y / scale
                
                # determine if inliers
                if (np.linalg.norm([pre_x - tar_x, pre_y - tar_y]) < sigma):
                    cnt += 1
                    tmp_mask[idx] = 1

            # update result
            if cnt > pre_total:  
                # update parameters
                pre_total = cnt
                t = cnt / count
                iters = np.log(1 - P) / np.log(1 - (1 - t) ** 4) 
                     
                # update result
                res_H = mat_H
                res_mask = tmp_mask

            # update iterations
            it += 1
        
        res_mask = np.int32(res_mask.reshape(count, 1))
        print("iter: ", it)
        return res_H, res_mask
    
    def doKernel(self, src_pt4, dst_pt4):
        src_len = src_pt4.shape[0]
        dst_len = dst_pt4.shape[0]
        
        if (src_len != 4 or dst_len != 4):
            return None    
        
        # do linear equations
        A = np.zeros((8, 8))
        b = dst_pt4.reshape((8, 1))
        for i in range(0, 4):
            A[2 * i, 0] = src_pt4[i, 0]
            A[2 * i, 1] = src_pt4[i, 1]
            A[2 * i, 2] = 1.0
            A[2 * i, 6] = -src_pt4[i, 0] * dst_pt4[i, 0]
            A[2 * i, 7] = -src_pt4[i, 1] * dst_pt4[i, 0]
            A[2 * i + 1, 3] = src_pt4[i, 0]
            A[2 * i + 1, 4] = src_pt4[i, 1]
            A[2 * i + 1, 5] = 1.0
            A[2 * i + 1, 6] = -src_pt4[i, 0] * dst_pt4[i, 1]
            A[2 * i + 1, 7] = -src_pt4[i, 1] * dst_pt4[i, 1]
        
        if (np.linalg.det(A) == 0):
            return None
        
        result = np.linalg.solve(A, b)
        mat_H = np.append(result, 1).reshape((3, 3))
        return mat_H
    
    def findHomography(self, src_pts, dst_pts, method, reproj_thresh = 3):
        src_len = src_pts.shape[0]
        dst_len = dst_pts.shape[0]
        
        if (src_len != dst_len or src_len < 4):
            return None, None
        
        if (reproj_thresh <= 0):
            reproj_thresh = 3
            
        if (method == cv.RANSAC):
            mat_H, mask = self.doRansac(src_pts, dst_pts, reproj_thresh)
            return mat_H, mask
        else:
            mat_H, mask = self.doRansac(src_pts, dst_pts, reproj_thresh)
            return mat_H, mask
        
        return None, None
        
    
# if __name__ == '__main__':
#     src_pts = np.array([[1, 2], [3, 4], [2, 5], [4, 6], [10, 5], [8, 9], [4, 2]])
#     dst_pts = np.array([[0, 1], [3, 3], [1, 8], [5, 10], [12, 6], [13, 18], [6, 8]])
#     hor = Homography()
#     H, status = hor.findHomography(src_pts, dst_pts, 0)
#     print(H)
#     print(status)
#     H, status = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
#     print(H)
#     print(status)