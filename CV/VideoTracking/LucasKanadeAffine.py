import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(6)
    x1,y1,x2,y2 = rect

    # fix the bounding box within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(It.shape[1], x2)
    y2 = min(It.shape[0], y2)

    I = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    T = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    X, Y = np.arange(x1,x2), np.arange(y1,y2)
    Tx = T(Y, X)
    xx, yy = np.meshgrid(X, Y)
    for iter in range(maxIters):
        x_warp = xx*(1+p[0]) + yy*p[1] + p[2]
        y_warp = xx*p[3] + yy*(1+p[4]) + p[5]
        Iy = I.ev(y_warp, x_warp, dx=1).flatten()
        Ix = I.ev(y_warp, x_warp, dy=1).flatten()
        y = y_warp.flatten()
        x = x_warp.flatten()
        J = np.vstack((x*Ix, y*Ix, Ix, x*Iy, y*Iy, Iy)).T
        b = (Tx - I.ev(y_warp, x_warp)).flatten()
        dp = np.linalg.lstsq(J, b)[0]
        p += dp
        if np.linalg.norm(dp) < threshold:
            break

    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
    return M
