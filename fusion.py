import numpy as np
import cv2 as cv
from utils import cv_show

class Fusion:
	
	# 泊松融合
	def poisson(self, result, image, reverse, t, h_top, w_top):
		h, w = image.shape[:2]
		for i in range(t[1], h_top):
			for j in range(t[0], w_top):
				if result[i][j][0] == 0 or result[i][j][1] == 0 or result[i][j][2] == 0:
					result[i][j] = image[i-t[1]][j-t[0]]
		mask = 255 * np.ones((h_top - t[1], w_top - t[0]), dtype=image.dtype)
		result = cv.seamlessClone(image, result, mask, (t[0] + (w_top - t[0]) // 2, t[1] + (h_top - t[1]) // 2), cv.NORMAL_CLONE)		
		return result


	# 加权融合
	def weigh_fussion(self, result, image, reverse, t, h_top, w_top):
		h, w = image.shape[:2]
		if(reverse == False):
			right_top, right_bottom = t[0], t[0]
			for j in range(w_top - 1, t[0] -1, -1):
				if result[t[1]][j][0] != 0 or result[t[1]][j][1] != 0 or result[t[1]][j][2] != 0:
					right_top = j
					break
			for j in range(w_top - 1, t[0] -1, -1):
				if result[h_top - 1][j][0] != 0 or result[h_top - 1][j][1] != 0 or result[h_top - 1][j][2] != 0:
					right_bottom = j
					break
			end = right_top if right_top > right_bottom else right_bottom
			width = end - t[0] + 1
			alpha = 1.0
			result[t[1]:h_top, end + 1:w_top] = image[0:h_top-t[1], width:w_top-t[0]]
			for i in range(t[1], h_top):
				for j in range(end, t[0] - 1, -1):
					if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
						alpha = 1.0
					else:
						alpha = 1.0 - (end - j) / width
					result[i][j] = alpha * image[i - t[1]][j - t[0]] + (1 - alpha) * result[i][j]
		else:
			left_top, left_bottom = w_top, w_top
			for j in range(t[0], w_top):
				if result[t[1]][j][0] != 0 or result[t[1]][j][1] != 0 or result[t[1]][j][2] != 0:
					left_top = j
					break
			for j in range(t[0], w_top):
				if result[h_top - 1][j][0] != 0 or result[h_top - 1][j][1] != 0 or result[h_top - 1][j][2] != 0:
					left_bottom = j
					break
			start = left_top if left_top < left_bottom else left_bottom
			width = w_top - start
			alpha = 1.0
			result[t[1]:h_top, t[0]:start] = image[0:h_top-t[1], 0:start - t[0]]
			for i in range(t[1], h_top):
				for j in range(start, w_top):
					if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
						alpha = 1.0
					else:
						alpha = 1.0 - (j - start) / width
					result[i][j] = alpha * image[i - t[1]][j - t[0]] + (1 - alpha) * result[i][j]
		return result

	# 金字塔融合
	def Multiband(self, result, image, reverse, t, h_top, w_top):
		num_levels = 6
		h, w = image.shape[:2]
		source = np.zeros(result.shape, dtype = result.dtype)
		source[t[1] : h_top, t[0] : w_top] = image[:h_top - t[1], :w_top - t[0]]
		mask = np.ones(result.shape, dtype = np.float32)
		mask[result == 0] = 0

		if (reverse == False):
			for i in range(t[1], h_top):
				right = t[0]
				for j in range(w_top - 1, t[0] -1, -1):
					if result[i][j][0] != 0 or result[i][j][1] != 0 or result[i][j][2] != 0:
						right = j
						break
				if right != t[0]:
					width = right - t[0]
					for j in range(t[0], right + 1):
						if(mask[i][j].all() == 1):
							mask[i][j] = (right - j) / width
		else:
			for i in range(t[1], h_top):
				left = w_top
				for j in range(t[0], w_top):
					if result[i][j][0] != 0 or result[i][j][1] != 0 or result[i][j][2] != 0:
						left = j
						break
				if left != w_top:
					width = w_top - left
					for j in range(left, w_top):
						if(mask[i][j].all() == 1):
							mask[i][j] = (j - left) / width

		# 生成高斯金字塔
		GA = result.copy()
		GB = source.copy()
		GM = mask.copy()
		gpA = [np.float32(GA)]
		gpB = [np.float32(GB)]
		gpM = [np.float32(GM)]
		for i in range(num_levels):
			GA = cv.pyrDown(GA)
			GB = cv.pyrDown(GB)
			GM = cv.pyrDown(GM)
			gpA.append(np.float32(GA))
			gpB.append(np.float32(GB))
			gpM.append(np.float32(GM))

		# 生成拉普拉斯金字塔
		lpA = [gpA[num_levels - 1]]
		lpB = [gpB[num_levels - 1]]
		gpMr = [gpM[num_levels - 1]]

		for i in range(num_levels - 1, 0, -1):
			size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
			LA = np.subtract(gpA[i - 1], cv.pyrUp(gpA[i], dstsize = size))
			LB = np.subtract(gpB[i - 1], cv.pyrUp(gpB[i], dstsize = size))
			lpA.append(LA)
			lpB.append(LB)
			gpMr.append(gpM[i - 1])

		# 生成融合金字塔
		lp = []
		for la, lb, gm in zip(lpA, lpB, gpMr):
			ls = la * gm + lb * (1.0 - gm)
			lp.append(ls)

		# 重建
		ls_ = lp[0]
		for i in range(1, num_levels):
			size = (lp[i].shape[1], lp[i].shape[0])
			ls_ = cv.pyrUp(ls_, dstsize = size)
			ls_ = cv.add(ls_, np.float32(lp[i]))
			ls_[ls_ > 255] = 255
			ls_[ls_ < 0] = 0

		ls_ = ls_.astype(np.uint8)
		# 融合结果		
		return ls_
		
	def simple(self, result, img, reverse, t, h_top, w_top):
		for i in range(t[1], h_top):
			for j in range(t[0], w_top):
				if result[i][j][0] == 0 or result[i][j][1] == 0 or result[i][j][2] == 0:
					result[i][j] = img[i-t[1]][j-t[0]]
		return result


	