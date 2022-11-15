import numpy as np
import pickle
from getImageFeatures import get_image_features
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# train
classifier = pickle.load(open('visionHarris.pkl', 'rb'))
trainFeatures = classifier['trainFeatures']
trainLabels = classifier['trainLabels'].flatten()
svm_linear = make_pipeline(StandardScaler(), SVC(kernel='linear'))
svm_linear.fit(trainFeatures, trainLabels)
svm_rbf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
svm_rbf.fit(trainFeatures, trainLabels)
classifier['svm_linear'] = svm_linear
classifier['svm_rbf'] = svm_rbf
with open(f'visionSVM.pkl', 'wb') as fh:
    pickle.dump(classifier, fh)

# test

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']

point_method = 'Harris'

classifier = pickle.load(open('visionSVM.pkl', 'rb'))
dictionary = classifier['dictionary']
svm_linear = classifier['svm_linear']
svm_rbf = classifier['svm_rbf']
predLabels_linear = np.empty((test_labels.shape))
predLabels_rbf = np.empty((test_labels.shape))

for i, test_imagename in enumerate(test_imagenames):
    wordMap = pickle.load(open('../data/%s_%s.pkl' % (test_imagename[:-4], point_method), 'rb'))
    testFeature = get_image_features(wordMap, len(dictionary))[None]
    predLabels_linear[i] = svm_linear.predict(testFeature)
    predLabels_rbf[i] = svm_rbf.predict(testFeature)

print('SVM with linear kernel')
acc = accuracy_score(test_labels, predLabels_linear)
print('accuracy:', acc)
cm = confusion_matrix(test_labels, predLabels_linear, labels=np.arange(1,9))
print('confusion matrix:')
print(cm)

print('-'*50)

print('SVM with rbf kernel')
acc = accuracy_score(test_labels, predLabels_rbf)
print('accuracy:', acc)
cm = confusion_matrix(test_labels, predLabels_rbf, labels=np.arange(1,9))
print('confusion matrix:')
print(cm)


# # ----------------------------------------------
