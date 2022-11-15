"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import scipy.optimize
import scipy.signal, scipy.linalg
import skimage
import helper

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):

    # normalize pts
    T = np.array([[1/M, 0, 0,], [0, 1/M, 0], [0, 0, 1]])
    pts1_h = np.insert(pts1, 2, 1, axis=1) @ T
    pts2_h = np.insert(pts2, 2, 1, axis=1) @ T

    # build matrix A and find F from SVD
    A = np.concatenate(((pts1_h[:,[0]] * pts2_h), (pts1_h[:,[1]] * pts2_h), (pts1_h[:,[2]] * pts2_h)), axis=1)
    u, s, vh = np.linalg.svd(A)
    F = vh[np.argmin(s)].reshape(3,3)

    # enforce rank 2 constraint
    u, s, vh = np.linalg.svd(F)
    s[np.argmin(s)] = 0.
    F_rank2 = u @ np.diag(s) @ vh

    # local minimization
    F_refined = helper.refineF(F_rank2, pts1_h[:,:2], pts2_h[:,:2])

    # un-normalize F
    F = T.T @ F_refined @ T

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    im1 = skimage.color.rgb2gray(im1)
    im2 = skimage.color.rgb2gray(im2)

    h1, w1 = im1.shape
    h2, w2 = im2.shape
    pts2 = []
    h, w = 5, 5
    im1_pad = np.pad(im1, ((h//2,h//2),(w//2,w//2)))
    im2_pad = np.pad(im2, ((h//2,h//2),(w//2,w//2)))
    for pt in pts1:
        x, y = pt.astype(int)
        pt_h = np.array([[x], [y], [1]])
        a, b, c = (F @ pt_h).reshape(-1)
        # ax + by + c = 0
        xs = np.arange(w2)
        ys = ((-a*xs - c)/b).astype(int)
        # dist = [(np.abs(
        #     (im2_pad[y2:y2+h,x2:x2+w] - im1_pad[y:y+h,x:x+w])
        #     )).sum()
        #     if y2 < h2 else np.inf for (y2,x2) in zip(ys,xs)]
        dist = [np.linalg.norm(im2_pad[y2:y2+h,x2:x2+w]-im1_pad[y:y+h,x:x+w])
            if y2 < h2 else np.inf for (y2,x2) in zip(ys,xs)]
        id = np.argmin(dist)
        pts2.append([xs[id], ys[id]])
    pts2 = np.array(pts2)
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    p1_1, p1_2, p1_3 = P1
    p2_1, p2_2, p2_3 = P2
    pts_3d = np.empty((pts1.shape[0], 4))
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        x1, y1 = pt1
        x2, y2 = pt2
        A = np.array([
            y1*p1_3 - p1_2,
            p1_1 - x1*p1_3,
            y2*p2_3 - p2_2,
            p2_1 - x2*p2_3,
        ])
        u, s, vh = np.linalg.svd(A)
        h = vh[np.argmin(s)]
        pts_3d[i] = h

    reproj_2 = pts_3d @ P2.T
    reproj_2 = reproj_2[:,:2] / reproj_2[:,[2]]

    reproj_1 = pts_3d @ P1.T
    reproj_1 = reproj_1[:,:2] / reproj_1[:,[2]]

    reprojection_error = np.linalg.norm(reproj_1-pts1, axis=1).mean()

    pts_3d = pts_3d[:,:3] / pts_3d[:,[3]]

    return pts_3d, reprojection_error


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    
    c1 = - np.linalg.inv(K1 @ R1) @ (K1 @ t1) # 3 x 1
    c2 = - np.linalg.inv(K2 @ R2) @ (K2 @ t2) # 3 x 1

    r1 = ((c1 - c2)/np.linalg.norm(c1 - c2))
    r2 = np.cross(R1[2], r1[:,0])[:,None]
    r3 = np.cross(r2[:,0], r1[:,0])[:,None]

    R_ = np.hstack((r1,r2,r3)).T

    R1p, R2p = R_, R_
    K1p, K2p = K2, K2
    t1p = -R_ @ c1
    t2p = -R_ @ c2

    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    
    def dist(i1, i2, d, win_size):
        i2_shifted = np.roll(i2, -d, axis=1)
        dist_m = (i1 - i2_shifted)**2
        w = (win_size - 1)//2
        return scipy.signal.convolve2d(dist_m, np.ones((win_size, win_size)))[w:-w, w:-w]

    disp = np.array([dist(im1, im2, d, win_size) for d in range(max_disp+1)])
    dispM = disp.argmin(axis=0)

    return dispM



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = - np.linalg.inv(K1 @ R1) @ (K1 @ t1) # 3 x 1
    c2 = - np.linalg.inv(K2 @ R2) @ (K2 @ t2) # 3 x 1

    b = np.linalg.norm(c1-c2)
    f = K1[0,0]

    depthM = np.where(dispM == 0, 0, b*f/dispM)

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    As = []
    for (x1, y1), (X1, Y1, Z1) in zip(x, X):
        A = np.array([
            [X1, Y1, Z1, 1, 0, 0, 0, 0, -x1*X1, -x1*Y1, -x1*Z1, -x1],
            [0, 0, 0, 0, X1, Y1, Z1, 1, -y1*X1, -y1*Y1, -y1*Z1, -y1]
        ])
        As.append(A)

    A = np.concatenate(As)

    u, s, vh = np.linalg.svd(A)
    h = vh[np.argmin(s)]
    P = h.reshape(3,4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    u, s, vh = np.linalg.svd(P)
    h = vh[np.argmin(s)]
    c = h[:3] / h[3]
    K, R = scipy.linalg.rq(P[:,:3])
    t = (-R@c)[:,None]

    return K, R, t

