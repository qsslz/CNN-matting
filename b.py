import numpy as np 
from scipy.misc import imread
import math
import random

def rgb2gray(image):
	result = np.zeros((image.shape[0], image.shape[1]))
	i = 0
	for row in image:
		j = 0                                                  
		for column in row:
			result[i][j] = (0.29900 * column[0] + 0.58700 * column[1] + 0.11400 * column[2]) / 255
			j += 1
		i += 1	
	return result

def reshape2rows(matrix):
	matrix = matrix.T;
	result = np.zeros((matrix.shape[0] * matrix.shape[1],1))
	index = 0
	i = 0
	j = 0
	for row in matrix:
		j = 0
		for column in row:
			result[index] = matrix[i][j]
			index += 1
			j += 1
		i += 1
	return result



def rgb2hsv(image):
	result = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
	i = 0
	for row in image:
		j = 0
		for column in row:
			r = column[0]/255.0
			g = column[1]/255.0
			b = column[2]/255.0
			mx = max(r, g, b)
			mn = min(r, g, b) 
			df = mx-mn
			if mx == mn:
   				h = 0
			elif mx == r:
				h = 60 * (g-b)/df
			elif mx == g:
				h = 60 * (b-r)/df + 120
			elif mx == b:
				h = 60 * (r-g)/df + 240
			if mx == 0:
				s = 0
			else:
				s = df/mx
			v = mx
			result[i][j][0] = h / 360
			result[i][j][1] = s
			result[i][j][2] = v
			j += 1
		i += 1
	return result

def knnmatting(im, scrib):
	lam = 1000  
	level = 2
	factor = 1
	im = rgb2hsv(im)
	scrib = reshape2rows(scrib)
	m = im.shape[0]
	n = im.shape[1]
	d = im.shape[2]	
	val = scrib > 0.99
	map_matting = (scrib < 0.01) + val
	nn =  np.array([[10],[2]])
	a = np.zeros((1, m * n))
	b = np.zeros((1, m * n))
	i = 1
	j = 1
	for j in range(1, n + 1):
		for i in range(1, m + 1):
			a[0][(j-1)*m+i-1]=i
			b[0][(j-1)*m+i-1]=j
			i =+ 1
		j =+ 1
	feature = np.zeros((6, m * n))
	feature[0, :] = reshape2rows(np.cos(im[:,:,0] * 2 * np.pi)).T * factor
	feature[1, :] = reshape2rows(np.sin(im[:,:,0] * 2 * np.pi)).T * factor
	feature[2, :] = reshape2rows(im[:,:,1]).T / 2
	feature[3, :] = reshape2rows(im[:,:,2]).T / 2
	feature[4, :] = a / np.sqrt(m*m+n*n)*level+ np.random.rand(1, m*n)*1e-6
	feature[5, :] = b / np.sqrt(m*m+n*n)*level+ np.random.rand(1, m*n)*1e-6
	now = 0

	
img = imread('/Users/qusong/Desktop/毕业设计/demo/demo/data/inputs/doll.png')
trimap = imread('/Users/qusong/Desktop/毕业设计/demo/demo/data/Trimap1/doll.png')
trimap = rgb2gray(trimap)
knnmatting(img, trimap)	


