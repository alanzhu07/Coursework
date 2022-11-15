import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h = np.bincount(wordMap.flatten(), minlength=dictionarySize)
    h = h / np.linalg.norm(h, 1)
    # ----------------------------------------------
    
    return h
