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
        min_iters = 300
        iters = 2000
        count = src_pts.shape[0]
        P = 0.95
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
        return res_mask
    
    def doKernel(self, src_pt4, dst_pt4):
        src_len = src_pt4.shape[0]
        dst_len = dst_pt4.shape[0]
        
        if (src_len < 4 or dst_len != src_len):
            return None
        
        # compute center of src and dst
        cs_x, cs_y, cd_x, cd_y = 0, 0, 0, 0
        for i in range(0, src_len):
            cs_x += src_pt4[i, 0]
            cs_y += src_pt4[i, 1]
            cd_x += dst_pt4[i, 0]
            cd_y += dst_pt4[i, 1]
        
        cs_x /= src_len
        cs_y /= src_len
        cd_x /= dst_len
        cd_y /= dst_len
        
        # remove center
        ss_x, ss_y, sd_x, sd_y = 0, 0, 0, 0
        for i in range(0, src_len):
            ss_x += np.fabs(src_pt4[i, 0] - cs_x)
            ss_y += np.fabs(src_pt4[i, 1] - cs_y)
            sd_x += np.fabs(dst_pt4[i, 0] - cd_x)
            sd_y += np.fabs(dst_pt4[i, 1] - cd_y)
        
        # check if zero
        EPS = np.finfo(float).eps
        if np.fabs(ss_x) < EPS or np.fabs(ss_y) < EPS or np.fabs(sd_x) < EPS or np.fabs(sd_y) < EPS:
            return None
        
        # calculate absolute error
        ss_x = src_len / ss_x
        ss_y = src_len / ss_y
        sd_x = dst_len / sd_x
        sd_y = dst_len / sd_y
        
        # scale matrixs
        mat_H = np.array([[ss_x, 0, -cs_x * ss_x], [0, ss_y, -cs_y * ss_y], [0, 0, 1]])
        mat_invH = np.array([[1.0 / sd_x, 0, cd_x], [0, 1.0 / sd_y, cd_y], [0, 0, 1]])
        mat_LtL = np.zeros((9, 9))
        
        for i in range(0, src_len):
            s_x = (src_pt4[i, 0] - cs_x) * ss_x
            s_y = (src_pt4[i, 1] - cs_y) * ss_y
            d_x = (dst_pt4[i, 0] - cd_x) * sd_x
            d_y = (dst_pt4[i, 1] - cd_y) * sd_y
            
            l_x = np.array([s_x, s_y, 1, 0, 0, 0, -s_x * d_x, -s_y * d_x, -d_x])
            l_y = np.array([0, 0, 0, s_x, s_y, 1, -s_x * d_y, -s_y * d_y, -d_y])
            
            for j in range(0, 9):
                for k in range(0, 9):
                    mat_LtL[j, k] += l_x[j] * l_x[k] + l_y[j] * l_y[k]
            
        # do SVD
        _, _, vh = np.linalg.svd(mat_LtL)        
        res_H = np.matmul(np.matmul(mat_invH, vh[8].reshape((3, 3))), mat_H)
        res_H = res_H * (1.0 / res_H[2, 2])
        return res_H
    
    def findHomography(self, src_pts, dst_pts, method, reproj_thresh = 3):
        src_len = src_pts.shape[0]
        dst_len = dst_pts.shape[0]
        
        if (src_len != dst_len or src_len < 4):
            return None, None
        
        if (reproj_thresh <= 0):
            reproj_thresh = 3
            
        # public mask
        mask = np.ones((src_len, 1), dtype=np.uint32)

        if (src_len == 4 or method == 0):
            mat_H = self.doKernel(src_pts, dst_pts)
            return mat_H, mask
            
        # use different methods
        if (method == cv.RANSAC):
            mask = self.doRansac(src_pts, dst_pts, reproj_thresh)
        else:
            print("Unknown method for homography! Use RANSAC instead.")
            mask = self.doRansac(src_pts, dst_pts, reproj_thresh)
        
        # find inliers
        src_in = np.array([])
        dst_in = np.array([])
        for idx in range(0, mask.shape[0]):
            if mask[idx] == 1:  
                src_in = np.append(src_in, src_pts[idx])
                dst_in = np.append(dst_in, dst_pts[idx])      
        
        src_in = src_in.reshape((src_in.shape[0] // 2, 2))
        dst_in = dst_in.reshape((dst_in.shape[0] // 2, 2))
    
        # calculate homography matrix
        mat_H = self.doKernel(src_in, dst_in)
        
        return mat_H, mask
        
    
# if __name__ == '__main__':
#     src_pts = np.array([[1, 2], [3, 4], [2, 5], [4, 6]])
#     dst_pts = np.array([[0, 1], [3, 3], [1, 8], [5, 10]])
# #     src_pts = np.array([[1, 2], [3, 4], [2, 5], [4, 6], [10, 5], [8, 9], [4, 2]])
# #     dst_pts = np.array([[0, 1], [3, 3], [1, 8], [5, 10], [12, 6], [13, 18], [6, 8]])
#     hor = Homography()
#     # H = hor.doKernel(src_pts, dst_pts)
#     H, status = hor.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
#     print(H)
#     print(status)
#     H, status = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
#     print(H)
#     print(status)