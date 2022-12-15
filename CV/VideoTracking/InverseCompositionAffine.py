import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
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
    Template = T(Y, X)
    xx, yy = np.meshgrid(X, Y)

    # pre-compute the Jacobian
    Ty = T.ev(yy, xx, dx=1).flatten()
    Tx = T.ev(yy, xx, dy=1).flatten()
    y = yy.flatten()
    x = xx.flatten()
    J = np.vstack((x*Tx, y*Tx, Tx, x*Ty, y*Ty, Ty)).T

    for iter in range(maxIters):
        x_warp = xx*(1+p[0]) + yy*p[1] + p[2]
        y_warp = xx*p[3] + yy*(1+p[4]) + p[5]
        b = (I.ev(y_warp, x_warp) - Template).flatten()
        dp = np.linalg.lstsq(J, b)[0]
        P = np.array([
            [1.0+p[0], p[1],    p[2]],
            [p[3],     1.0+p[4], p[5]],
            [0, 0, 1]])
        DP = np.array([
            [1.0+dp[0], dp[1],    dp[2]],
            [dp[3],     1.0+dp[4], dp[5]],
            [0, 0, 1]])
        p = ((P @ np.linalg.inv(DP)) - np.eye(3)).flatten()[:6]
        if np.linalg.norm(dp) < threshold:
            break


    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
