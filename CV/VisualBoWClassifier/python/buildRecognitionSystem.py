import pickle
import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features

# -----fill in your implementation here --------

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']
trainLabels = meta['train_labels'][:, None]
filterBank = create_filterbank()

for point_method in ['Random', 'Harris']:
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    trainFeatures = np.empty((len(train_imagenames), len(dictionary)))
    for i, path in enumerate(train_imagenames):
        print('-- processing %d/%d' % (i, len(train_imagenames)))
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (path[:-4], point_method), 'rb'))
        trainFeatures[i] = get_image_features(wordMap, len(dictionary))
    classifier = {
        'dictionary': dictionary,
        'filterBank': filterBank,
        'trainFeatures': trainFeatures,
        'trainLabels': trainLabels
    }
    with open(f'vision{point_method}.pkl', 'wb') as fh:
        pickle.dump(classifier, fh)
    print(point_method, 'done')

# ---------------------------------
# -------------
