import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *

from createFilterBank import create_filterbank


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------

    img_lab = rgb2lab(I)
    filterResponses = []
    for filter in filterBank:
        filtered = imfilter(img_lab, filter[..., None])
        filterResponses.append(filtered)
    # for filter in filterBank:
    #     for i in range(3):
    #         filtered = imfilter(img_lab[..., i], filter)
    #         filterResponses.append(filtered)
    filterResponses = np.dstack(filterResponses)
    # print(filterResponses.shape)

    # ----------------------------------------------
    return filterResponses

if __name__ == '__main__':
    im1 = cv.cvtColor(cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg'), cv.COLOR_BGR2RGB)
    cv.imwrite("../output/original.jpg", im1)
    filterbank = create_filterbank()
    print(len(filterbank))
    filterResponses = extract_filter_responses(im1, filterbank)
    print(filterResponses[:5,:5,0])
    print(filterResponses[:5,:5,1])
    for i in range(20):
        response = filterResponses[:,:,i*3:(i+1)*3]
        response = 255 * (response - response.min()) / response.ptp()
        response = cv.cvtColor(response.astype(np.uint8), cv.COLOR_BGR2GRAY)
        cv.imwrite(f"../output/img_{i}.jpg", response)
