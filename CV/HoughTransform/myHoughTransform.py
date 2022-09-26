import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    thetaScale = np.arange(0, 2*np.pi, thetaRes)
    rhoMax = np.sqrt(Im.shape[0]**2 + Im.shape[1]**2)
    rhoScale = np.arange(0, rhoMax, rhoRes)

    img_hough = np.zeros((rhoScale.shape[0], thetaScale.shape[0]))

    y = np.arange(Im.shape[0])[:,None]
    x = np.arange(Im.shape[1])[None,:]
    for j, theta in enumerate(thetaScale):
        rho = (x*np.cos(theta) + y*np.sin(theta))
        rho = rho[(rho >= 0) & (Im == 1)]
        i = np.abs(rho[None,:] - rhoScale[:,None]).argmin(axis=0)

        for i_ in i:
            img_hough[i_, j] += 1

    return img_hough, rhoScale, thetaScale


