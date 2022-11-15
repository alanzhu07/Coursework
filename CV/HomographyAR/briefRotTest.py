import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
import skimage.color
import scipy.ndimage
import matplotlib.pyplot as plt

#Q3.5
#Read the image and convert to grayscale, if necessary

cv_cover = cv2.imread('../data/cv_cover.jpg')
img = skimage.color.rgb2gray(cv_cover)

num_matches = np.empty(36)
for i in range(36):
	#Rotate Image
	img_rot = scipy.ndimage.rotate(img, i*10)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, img_rot)

	#Update histogram
	num_matches[i] = matches.shape[0]
	print(f'rotation degree: {i*10}, matches: {num_matches[i]}')

	if i == 1 or i == 9 or i == 20: 
		cv_cover_rot = scipy.ndimage.rotate(cv_cover, i*10)
		plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)

#Display histogram
plt.bar(np.arange(0, 360, 10), num_matches, width=10)
plt.xlabel("Rotation degree")
plt.ylabel("Number of matches")
plt.savefig('../output/q3_5.jpg')

