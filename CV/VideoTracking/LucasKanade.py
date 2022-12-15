import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1,y1,x2,y2 = rect

    # put your implementation here
    I = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    T = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    X, Y = np.arange(x1,x2), np.arange(y1,y2)
    for iter in range(maxIters):
        Y_warp = Y+p[1]
        X_warp = X+p[0]
        Iy = I(Y_warp, X_warp, dx=1).flatten()
        Ix = I(Y_warp, X_warp, dy=1).flatten()
        J = np.vstack((Ix, Iy)).T
        b = T(Y, X).flatten() - I(Y_warp, X_warp).flatten()
        dp = np.linalg.lstsq(J, b)[0]
        p += dp
        if np.linalg.norm(dp) < threshold:
            break

    return p

