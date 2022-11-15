import numpy as np
from itertools import product
import pickle
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# -----fill in your implementation here --------

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']

point_method = 'Harris'
dist_method = 'chi2'

ks = np.arange(1, 41)
accuracies = np.empty(ks.shape)
cms = []
for idx, k in enumerate(ks):
    classifier = pickle.load(open('vision%s.pkl' % point_method, 'rb'))
    dictionary = classifier['dictionary']
    trainFeatures = classifier['trainFeatures']
    trainLabels = classifier['trainLabels']
    predLabels = np.empty((test_labels.shape))

    for i, test_imagename in enumerate(test_imagenames):
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (test_imagename[:-4], point_method), 'rb'))
        testFeature = get_image_features(wordMap, len(dictionary))[None]
        dist = get_image_distance(testFeature, trainFeatures, dist_method).flatten()
        # print(dist.shape)
        k_nearests = np.argpartition(dist, k)[:k]
        predLabels[i] = np.bincount(trainLabels[k_nearests].flatten().astype(int)).argmax()

    accuracies[idx] = accuracy_score(test_labels, predLabels)
    cm = confusion_matrix(test_labels, predLabels, labels=np.arange(1,9))
    cms.append(cm)

best_accuracy = accuracies.max()
best_k = ks[accuracies.argmax()]
best_k_cm = cms[accuracies.argmax()]
print('best k =', best_k)
print('accuracy:', best_accuracy)
print('confusion matrix:')
print(best_k_cm)

plt.plot(ks, accuracies)
plt.xlabel('k in k-NN')
plt.ylabel('Accuracy')
plt.savefig('../output/plot.png')
