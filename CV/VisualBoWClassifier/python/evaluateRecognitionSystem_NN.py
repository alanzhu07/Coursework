import numpy as np
from itertools import product
import pickle
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance
from sklearn.metrics import confusion_matrix, accuracy_score

# -----fill in your implementation here --------

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']

point_methods = ['Random', 'Harris']
dist_methods = ['euclidean', 'chi2']

for point_method, dist_method in product(point_methods, dist_methods):
    print('_'*30)
    print(point_method, dist_method)

    classifier = pickle.load(open('vision%s.pkl' % point_method, 'rb'))
    dictionary = classifier['dictionary']
    trainFeatures = classifier['trainFeatures']
    trainLabels = classifier['trainLabels']
    predLabels = np.empty((test_labels.shape))

    for i, test_imagename in enumerate(test_imagenames):
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (test_imagename[:-4], point_method), 'rb'))
        testFeature = get_image_features(wordMap, len(dictionary))[None]
        dist = get_image_distance(testFeature, trainFeatures, dist_method)
        predLabels[i] = trainLabels[dist.argmin()][0]

    acc = accuracy_score(test_labels, predLabels)
    print('accuracy:', acc)
    cm = confusion_matrix(test_labels, predLabels, labels=np.arange(1,9))
    print('confusion matrix:')
    print(cm)


# ----------------------------------------------
