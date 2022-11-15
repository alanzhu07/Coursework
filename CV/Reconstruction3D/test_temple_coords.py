import numpy as np
import cv2
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

im1 = cv2.cvtColor(cv2.imread('../data/im1.png'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('../data/im2.png'), cv2.COLOR_BGR2RGB)

# 2. Run eight_point to compute F
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)
print('F:\n', F)
# hlp.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load('../data/temple_coords.npz')
temple_pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
temple_pts2 = sub.epipolar_correspondences(im1, im2, F, temple_pts1)
# hlp.epipolarMatchGUI(im1, im2, F)

# 5. Compute the camera projection matrix P1
data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']
E = sub.essential_matrix(F, K1, K2)
# print('Essential Matrix:\n', E)
P1 = K1 @ np.eye(3,4)

# 6. Use camera2 to get 4 camera projection matrices P2
extrinsic_candidates = hlp.camera2(E)
P2s = [K2 @ extrinsic_candidates[:,:,i] for i in range(4)]

# 7. Run triangulate using the projection matrices
pts_3ds = []
num_positive_zs = []
for i in range(4):
    pts_3d, _ = sub.triangulate(P1, temple_pts1, P2s[i], temple_pts2)
    # _, reprojection_error = sub.triangulate(P1, pts1, P2s[i], pts2)
    # print('reprojection error', reprojection_error)

    num_positive_zs.append((pts_3d[:,2] > 0).sum())
    pts_3ds.append(pts_3d)

# 8. Figure out the correct P2
P2 = P2s[np.argmax(num_positive_zs)]
pts_3d = pts_3ds[np.argmax(num_positive_zs)]


# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
E1 = np.eye(3,4)
E2 = extrinsic_candidates[:,:,np.argmax(num_positive_zs)]
R1, t1 = E1[:,:3], E1[:,[3]]
R2, t2 = E2[:,:3], E2[:,[3]]
np.savez('../data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)