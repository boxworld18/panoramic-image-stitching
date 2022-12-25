import numpy as np
import cv2 as cv
from utils import cv_show

class Fusion:
	
	# 泊松融合
	def poisson(self, result, imageB):
		for i in range(imageB.shape[0]):
			for j in range(imageB.shape[1]):
				if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
					result[i][j] = imageB[i][j]		
		mask = 255 * np.ones(imageB.shape, dtype=imageB.dtype)
		result = cv.seamlessClone(imageB, result, mask, (imageB.shape[1] // 2, imageB.shape[0] // 2), cv.NORMAL_CLONE)
		return result

	# 加权融合
	def weigh_fussion(self, result, imageB):
		left_top, left_bottom = imageB.shape[1], imageB.shape[1]
		for j in range(result.shape[1]):
			if result[0][j][0] != 0 or result[0][j][1] != 0 or result[0][j][2] != 0:
				left_top = j
				break
		for j in range(result.shape[1]):
			if result[imageB.shape[0] - 1][j][0] != 0 or result[imageB.shape[0] - 1][j][1] != 0 or result[imageB.shape[0] - 1][j][2] != 0:
				left_bottom = j
				break
		start = left_top if left_top < left_bottom else left_bottom
		width = imageB.shape[1] - start
		alpha = 1.0
		result[0:imageB.shape[0], 0:start] = imageB[0:imageB.shape[0], 0:start]
		for i in range(imageB.shape[0]):
			for j in range(start, imageB.shape[1]):
				if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
					alpha = 1.0
				else:
					alpha = 1.0 - (j - start) / width
				result[i][j] = alpha * imageB[i][j] + (1 - alpha) * result[i][j]
		return result

	# 金字塔融合
	def Multiband(self, result, imageB):
		# 计算重叠区域边界
		left_top, left_bottom = imageB.shape[1], imageB.shape[1]
		for j in range(result.shape[1]):
			if result[0][j][0] != 0 or result[0][j][1] != 0 or result[0][j][2] != 0:
				left_top = j
				break
		for j in range(result.shape[1]):
			if result[imageB.shape[0] - 1][j][0] != 0 or result[imageB.shape[0] - 1][j][1] != 0 or result[imageB.shape[0] - 1][j][2] != 0:
				left_bottom = j
				break
		start = left_top if left_top < left_bottom else left_bottom

		imageA = result[0 : imageB.shape[0], start : imageB.shape[1]]
		result[0 : imageB.shape[0], 0 : start] = imageB[0 : imageB.shape[0], 0 : start]
		imageB = imageB[0 : imageB.shape[0], start : imageB.shape[1]]

		num_levels = 6
		mask = np.ones(imageA.shape, dtype = imageA.dtype)
		mask[imageA == 0] = 0

		# 生成高斯金字塔
		GA = imageA.copy()
		GB = imageB.copy()
		GM = mask.copy()
		gpA = [GA]
		gpB = [GB]
		gpM = [GM]
		for i in range(num_levels):
			GA = cv.pyrDown(GA)
			GB = cv.pyrDown(GB)
			GM = cv.pyrDown(GM)
			gpA.append(GA)
			gpB.append(GB)
			gpM.append(GM)

		# 生成拉普拉斯金字塔
		lpA = [gpA[num_levels - 1]]
		lpB = [gpB[num_levels - 1]]
		gpMr = [gpM[num_levels - 1]]

		for i in range(num_levels - 1, 0, -1):
			size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
			LA = cv.subtract(gpA[i - 1], cv.pyrUp(gpA[i], dstsize = size))
			LB = cv.subtract(gpB[i - 1], cv.pyrUp(gpB[i], dstsize = size))
			lpA.append(LA)
			lpB.append(LB)
			gpMr.append(gpM[i - 1])

		# 生成融合金字塔
		lp = []
		for la, lb, gm in zip(lpA, lpB, gpMr):
			# ls = la * gm + lb * (1 - gm)
			ls = cv.addWeighted(la, 0.5, lb, 0.5, 0.0)
			lp.append(ls)

		# 重建
		ls_ = lp[0]
		for i in range(1, num_levels):
			size = (lp[i].shape[1], lp[i].shape[0])
			ls_ = cv.pyrUp(ls_, dstsize = size)
			ls_ = cv.add(ls_, lp[i])
			ls_[ls_ > 255] = 255
			ls_[ls_ < 0] = 0

		# 融合结果		
		result[0 : imageB.shape[0], start : start + ls_.shape[1]] = ls_
		return result
		


	