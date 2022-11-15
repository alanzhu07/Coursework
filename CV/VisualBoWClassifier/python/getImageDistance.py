import numpy as np
from scipy.spatial.distance import cdist
from utils import chi2dist

def get_image_distance(hist1, histSet, method):

    # -----fill in your implementation here --------
    if method == 'chi2':
        dist = chi2dist(hist1, histSet)
    elif method == 'euclidean':
        dist = cdist(hist1, histSet, 'euclidean')
    else:
        raise Exception('distance method not valid')

    # ----------------------------------------------

    return dist
