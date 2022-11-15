import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        filter_responses = extract_filter_responses(image, filterBank)
        if method.lower() == 'random':
            points = get_random_points(image, alpha)
        else:
            points = get_harris_points(image, alpha, k=0.05)
        x, y = points[:,0], points[:,1]
        pixelResponses[alpha*i:alpha*(i+1)] = filter_responses[y, x]


        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary
