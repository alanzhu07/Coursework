import numpy as np
import math

########################################################################
#######  you should maintain the return type in starter codes   #######
########################################################################


def perceptron_predict(w, x):
  # Input:
  #   w is the weight vector (d,),  1-d array
  #   x is feature values for test example (d,), 1-d array
  # Output:
  #   the predicted label for x, scalar -1 or 1

  _temp = w.dot(x)
  if _temp <= 0:
    return -1
  else:
    return 1


def perceptron_train(w0, XTrain, yTrain, num_epoch):
  # Input:
  #   w0 is the initial weight vector (d,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  # Output:
  #   the trained weight vector, (d,), 1-d array

  n = XTrain.shape[0]
  w = w0
  for k in range(num_epoch):
    for i in range(n):
      y_hat = perceptron_predict(w, XTrain[i])
      if y_hat != yTrain[i]:
        w = w + yTrain[i] * XTrain[i]
  return w


def RBF_kernel(X1, X2, sigma):
  # Input:
  #   X1 is a feature matrix (n,d), 2-d array or 1-d array (d,) when n = 1
  #   X2 is a feature matrix (m,d), 2-d array or 1-d array (d,) when m = 1
    #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   K is a kernel matrix (n,m), 2-d array

  if len(X1.shape) == 2:
    n = X1.shape[0]
  else:
    n = 1
    X1 = np.reshape(X1, (1, X1.shape[0]))
  if len(X2.shape) == 2:
    m = X2.shape[0]
  else:
    m = 1  
    X2 = np.reshape(X2, (1, X2.shape[0]))
  func = lambda i, j: np.exp(-np.square(np.linalg.norm(X1[i] - X2[j]))/(2*np.square(sigma)))
  K = np.fromfunction(np.vectorize(func), (n,m), dtype=int)
  return K

def RBF_kernel_opti(x, X2, sigma):
  return np.exp(-np.square(np.linalg.norm(x - X2, axis=1))/(2*np.square(sigma)))

def kernel_perceptron_predict_opti(a, XTrain, yTrain, x, sigma):
  val = np.sum(RBF_kernel_opti(x, XTrain, sigma) * a * yTrain)
  return -1 if val <= 0 else 1

def kernel_perceptron_predict(a, XTrain, yTrain, x, sigma):
  # Input:
  #   a is the counting vector (n,),  1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   x is feature values for test example (d,), 1-d array
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the predicted label for x, scalar -1 or 1

  x = np.reshape(x, (1, x.shape[0]))
  a = np.reshape(a, (1, a.shape[0]))
  yTrain = np.reshape(yTrain, (1, yTrain.shape[0]))
  val = np.sum(RBF_kernel(x, XTrain, sigma) * a * yTrain)
  return -1 if val <= 0 else 1
 
def kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma):
  # Input:
  #   a0 is the initial counting vector (n,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the trained counting vector, (n,), 1-d array
  
  a = a0
  n = a0.shape[0]
  for k in range(num_epoch):
    for i in range(n):
      y_i_hat = kernel_perceptron_predict_opti(a, XTrain, yTrain, XTrain[i], sigma)
      a[i] = a[i] + 1 if y_i_hat != yTrain[i] else a[i]
  return a