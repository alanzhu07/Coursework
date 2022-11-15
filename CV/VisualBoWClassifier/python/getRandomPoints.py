import numpy as np
import cv2 as cv

def get_random_points(I, alpha):

    # -----fill in your implementation here --------

    y = np.random.choice(I.shape[0], alpha)
    x = np.random.choice(I.shape[1], alpha)
    points = np.vstack((x,y)).T

    # ----------------------------------------------

    return points

if __name__ == '__main__':
    im1 = cv.cvtColor(cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg'), cv.COLOR_BGR2RGB)
    # dx, dy = get_harris_points(im1, 2, 3)
    # cv.imwrite(f"../output/dx.jpg", 255 * dx / dx.max())
    # cv.imwrite(f"../output/dy.jpg", 255 * dy / dy.max())
    points = get_random_points(im1, 1000)
    print(points.shape)
    # print(points)
    print(points[:,0].max(), points[:,1].max())