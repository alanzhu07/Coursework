import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    dx = ndimage.sobel(I, 0)
    dy = ndimage.sobel(I, 1)
    dxx = dx*dx
    dxy = dx*dy
    dyy = dy*dy

    # h = np.ones((5,5))
    # Sxx = imfilter(dxx, h)
    # Sxy = imfilter(dxy, h)
    # Syy = imfilter(dyy, h)
    Sxx = ndimage.gaussian_filter(dxx, 1)
    Sxy = ndimage.gaussian_filter(dxy, 1)
    Syy = ndimage.gaussian_filter(dyy, 1)
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    # print(det, trace)
    R = det - k * (trace ** 2)

    # print(R.shape)
    alpha_largest = np.argpartition(R, -alpha, axis=None)[-alpha:]
    y, x = np.unravel_index(alpha_largest, R.shape)
    points = np.vstack((x,y)).T

    # ----------------------------------------------
    
    return points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from getRandomPoints import get_random_points
    im = cv.cvtColor(cv.imread('../data/bedroom/sun_aacyfyrluprisdrx.jpg'), cv.COLOR_BGR2RGB)

    points_harris = get_harris_points(im, 500, 0.05)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_harris[:,0], points_harris[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/harris_im1.jpg')

    points_random = get_random_points(im, 500)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_random[:,0], points_random[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/random_im1.jpg')

    im = cv.cvtColor(cv.imread('../data/landscape/sun_aawnncfvjepzpmly.jpg'), cv.COLOR_BGR2RGB)

    points_harris = get_harris_points(im, 500, 0.05)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_harris[:,0], points_harris[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/harris_im2.jpg')

    points_random = get_random_points(im, 500)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_random[:,0], points_random[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/random_im2.jpg')

    im = cv.cvtColor(cv.imread('../data/campus/sun_abslhphpiejdjmpz.jpg'), cv.COLOR_BGR2RGB)

    points_harris = get_harris_points(im, 500, 0.05)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_harris[:,0], points_harris[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/harris_im3.jpg')

    points_random = get_random_points(im, 500)
    plt.clf()
    plt.imshow(im)
    plt.scatter(points_random[:,0], points_random[:,1], s=0.5, marker="*", c="red")
    plt.savefig('../output/random_im3.jpg')