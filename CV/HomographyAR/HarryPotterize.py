import numpy as np
import cv2
import skimage.io 
import skimage.color
import skimage.transform
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q3.9
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = skimage.transform.resize(hp_cover, (cv_cover.shape[0], cv_cover.shape[1]), anti_aliasing=True)

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=0.1, ratio=0.65)
x_cv_cover = locs1[matches[:,0]][:,[1,0]]
x_cv_desk = locs2[matches[:,1]][:,[1,0]]
H_cover_to_desk, inliers = computeH_ransac(x_cv_desk, x_cv_cover, num_iters=500, threshold=1.5)
composite = compositeH(H_cover_to_desk, cv_desk, hp_cover)

print(f" number of matches {matches.shape[0]}, number of inliers {inliers.sum()}")
cv2.imwrite("img.jpg", composite)