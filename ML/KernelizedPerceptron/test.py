import os
import csv
import numpy as np
import perceptron
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('.','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# Visualize the image
# idx = 0
# datapoint = XTrain[idx, 1:]
# plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
# plt.show()

# Non-Kernelized
# num_epoch = 20
# w0 = np.zeros((XTrain.shape[1]))
# w = perceptron.perceptron_train(w0, XTrain, yTrain, num_epoch)
# y = np.apply_along_axis(lambda x:perceptron.perceptron_predict(w, x), 1, XTest)
# test_error = np.sum(y != yTest)/y.shape[0]*100
# print("Test error:", test_error)

# Kernelized
num_epoch = 20
sigmas = [100,200]
a0 = np.zeros((yTrain.shape[0]))
for sigma in sigmas:
    a = perceptron.kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma)
    y = np.apply_along_axis(lambda x:perceptron.kernel_perceptron_predict(a, XTrain, yTrain, x, sigma), 1, XTest)
    test_error = np.sum(y != yTest)/y.shape[0]*100
    print("Sigma = {}, test error = {}".format(sigma, test_error))