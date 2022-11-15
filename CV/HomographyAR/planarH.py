import numpy as np
import cv2
import skimage.transform


def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points

	Ais = []
	for (x_1, y_1), (x_2, y_2) in zip(x1, x2):
		A_i = np.array(
			[
				[-x_2, -y_2, -1, 0, 0, 0, x_2*x_1, y_2*x_1, x_1],
				[0, 0, 0, -x_2, -y_2, -1, x_2*y_1, y_2*y_1, y_1]
			]
		)
		Ais.append(A_i)

	A = np.concatenate(Ais)
	u, s, vh = np.linalg.svd(A)
	h = vh[np.argmin(s)]
	H2to1 = h.reshape(3,3)

	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	x1_mean = x1.mean(axis=0)
	x2_mean = x2.mean(axis=0)

	#Shift the origin of the points to the centroid
	x1_ = x1 - x1_mean
	x2_ = x2 - x2_mean

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1_scale = 1 / np.abs(x1_).max(axis=0)
	x2_scale = 1 / np.abs(x2_).max(axis=0)

	#Similarity transform 1
	T_1 = np.array(
		[
			[x1_scale[0], 0, -x1_mean[0]*x1_scale[0]],
			[0, x1_scale[1], -x1_mean[1]*x1_scale[1]],
			[0, 0, 1]
		]
	)
	x1_homogeneous = np.append(x1, np.ones((len(x1), 1)), axis=1)
	x1_transformed = (x1_homogeneous @ T_1.T)[:,:2]

	#Similarity transform 2
	T_2 = np.array(
		[
			[x2_scale[0], 0, -x2_mean[0]*x2_scale[0]],
			[0, x2_scale[1], -x2_mean[1]*x2_scale[1]],
			[0, 0, 1]
		]
	)
	x2_homogeneous = np.append(x2, np.ones((len(x2), 1)), axis=1)
	x2_transformed = (x2_homogeneous @ T_2.T)[:,:2]

	#Compute homography
	H = computeH(x1_transformed, x2_transformed)

	#Denormalization
	H2to1 = np.linalg.inv(T_1) @ H @ T_2

	return H2to1




def computeH_ransac(x1, x2, num_iters=100, threshold=2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points

	best_inliers_count = 0
	x2_homogeneous = np.append(x2, np.ones((len(x2), 1)), axis=1)
	for _ in range(num_iters):
		points = np.random.choice(len(x1), 4, replace=False)
		x1_sampled, x2_sampled = x1[points], x2[points]
		H_ransac = computeH_norm(x1_sampled, x2_sampled)

		x1_transformed_homogeneous = x2_homogeneous @ H_ransac.T
		x1_transformed = x1_transformed_homogeneous[:,:2] / x1_transformed_homogeneous[:,[2]]
		inliers = (np.linalg.norm(x1_transformed - x1, axis=1) < threshold)

		if inliers.sum() > best_inliers_count:
			best_inliers_count = inliers.sum()
			best_inliers = inliers

	# recompute H using inliers
	to_select = best_inliers.nonzero()[0]
	x1_selected, x2_selected = x1[to_select], x2[to_select]
	bestH2to1 = computeH_norm(x1_selected, x2_selected)
	x1_transformed_homogeneous = x2_homogeneous @ bestH2to1.T
	x1_transformed = x1_transformed_homogeneous[:,:2] / x1_transformed_homogeneous[:,[2]]
	inliers = (np.linalg.norm(x1_transformed - x1, axis=1) < threshold)
	# print(best_inliers_count, inliers.sum())

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H_inv = np.linalg.inv(H2to1)

	warped = skimage.transform.warp(img, H_inv, output_shape=(template.shape[0], template.shape[1]))
	warped = (warped*255).astype(np.int_)

	mask = np.ones_like(img)
	mask = (skimage.transform.warp(mask, H_inv, output_shape=(template.shape[0], template.shape[1])) <= 0).astype(np.int_)

	composite_img = mask*template + warped

	return composite_img


