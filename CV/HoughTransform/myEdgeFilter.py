import numpy as np
from scipy import signal    # For signal.gaussian function
import cv2

from myImageFilter import myImageFilter

def myBlur(img0, sigma):
    hsize = int(2 * np.ceil(3 * sigma) + 1)
    kern = signal.windows.gaussian(hsize, sigma)
    kern2d = np.outer(kern, kern)
    return myImageFilter(img0, kern2d)

def myEdgeFilter(img0, sigma):
    img_blur = myBlur(img0, sigma)
    sobel_x = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    sobel_y = sobel_x.T
    imgx, imgy = myImageFilter(img_blur, sobel_x), myImageFilter(img_blur, sobel_y)
    magnitude = np.sqrt(imgx*imgx + imgy*imgy)
    direction = (np.arctan2(imgy, imgx) * 180. / np.pi) % 180.
    return nms_gradient(magnitude, direction)

def group(magnitude, direction):
    degrees = np.array([0., 45., 90., 135., 180.])
    gradient_grouped = np.abs(direction[None,:,:] - degrees[:,None,None]).argmin(axis=0) % 4
    return gradient_grouped

def nms_gradient(magnitude, direction):
    dilate_kernels = np.array([
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
 
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0]],
        
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]
    ], dtype=np.uint8)

    mask = group(magnitude, direction)

    output = np.zeros(magnitude.shape)
    for i in range(4):
        dilated = cv2.dilate(magnitude, dilate_kernels[i])
        output += np.where((dilated > magnitude) | (mask != i), 0., magnitude) 
    
    return output / output.max()