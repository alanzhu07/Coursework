import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    H_flat = nms(H).flatten()
    order = H_flat.argpartition(-nLines)[-nLines:]
    rhos, thetas = np.unravel_index(order, H.shape)
    return rhos, thetas

def nms(input):
    dilate_kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(input, dilate_kernel)
    output = np.where(dilated > input, 0, input)
    return output