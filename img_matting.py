import numpy as np
from scipy.misc import imread

def rgb2gray(image):
	result = np.zeros((image.shape[0], image.shape[1]))
	i = 0
	j = 0
	for row in image:
		j = 0
		for column in row:
			result[i][j] = (0.29900 * column[0] + 0.58700 * column[1] + 0.11400 * column[2]) / 255
			j += 1
		i += 1	
	return result


img = imread('E:\download\qq\demo_dcnnmatting\demo\data\inputs\doll.png')
trimap = imread('E:\download\qq\demo_dcnnmatting\demo\data\Trimap1\doll.png')

print(img.size)

if(trimap.shape[2] == 3):
	trimap = rgb2gray(trimap)

print(trimap.size)	
