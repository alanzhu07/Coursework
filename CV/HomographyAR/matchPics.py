import numpy as np
import cv2
import skimage.color
import skimage.feature
from helper import computeBrief

# modified helper function
def briefMatch(desc1,desc2,ratio=0.8):

	matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
	return matches

# modified helper function
PATCHWIDTH = 9
def corner_detection(im, sigma=0.15):
	# fast method
	result_img = skimage.feature.corner_fast(im, PATCHWIDTH, sigma)
	locs = skimage.feature.corner_peaks(result_img, min_distance=1)
	return locs

def matchPics(I1, I2, sigma=0.15, ratio=0.65):
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	if I1.ndim == 3:
		I1 = skimage.color.rgb2gray(I1)
	if I2.ndim == 3:
		I2 = skimage.color.rgb2gray(I2)
	
	#Detect Features in Both Images
	locs1 = corner_detection(I1, sigma=sigma)
	locs2 = corner_detection(I2, sigma=sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1, locs1)
	desc2, locs2 = computeBrief(I2, locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio=ratio)

	return matches, locs1, locs2
