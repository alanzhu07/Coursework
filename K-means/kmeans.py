import numpy as np
import collections
import math
from scipy.io import loadmat

def update_assignments(X, C):
  """
    Arguments:
      X: (N, D) numpy array
      C: (K, D) numpy array
    Returns:
      assignments: (N,) numpy array
  """
  dist = lambda i: np.argmin(np.linalg.norm(C-X[i],axis=1))
  assignments = np.array([dist(i) for i in range(X.shape[0])])
  return assignments

def update_centers(X, prev_C, assignments):
  """
    Arguments:
      X: (N, D) numpy array
      prev_C: (K, D) numpy array
      assignments: (N,) numpy array
    Returns:
      C: (K, D) numpy array
  """
  cluster_center = lambda i: np.mean(X[assignments==i], axis=0)
  C = np.array([cluster_center(i) for i in range(prev_C.shape[0])])
  return C

def lloyd_iteration(X, C0):
  """
    Arguments:
      X: (N, D) numpy array
      C0: (K, D) numpy array
    Returns:
      C: (K, D) numpy array
      assignments: (N,) numpy array
  """
  C = C0 # initial center
  assignments = None
  while True:
    assignments_new = update_assignments(X, C) # form clusters
    if np.array_equal(assignments, assignments_new):
      break # reached convergence
    else:
      assignments = assignments_new
      C = update_centers(X, C, assignments) # recalculate centers
  return C, assignments

def kmeans_obj(X, C, assignments):
  """
    Arguments:
      X: (N, D) numpy array
      C: (K, D) numpy array
      assignments: (N,) numpy array
    Returns:
      obj: a float
  """
  kmeans_dist = lambda i : np.linalg.norm(X[i] - C[assignments[i]])**2
  obj = np.sum([kmeans_dist(i) for i in range(X.shape[0])])
  return obj

def discrete_sample(weights):
  weights = weights / weights.sum()
  return np.random.choice(weights.shape[0], 1, p=weights)[0]

def kmeanspp_init(X, k):
  n = X.shape[0]
  sq_min_dist = np.ones((n,), dtype="float32") * 10000
  C = np.zeros((k, X.shape[1]), dtype="float32")
  for i in range(k):
    idx = discrete_sample(sq_min_dist)
    C[i] = X[idx]
    sq_dist = np.power(X - X[idx: idx + 1], 2).sum(axis=1)
    sq_min_dist = np.minimum(sq_min_dist, sq_dist)
  return C

def kmeans_cluster(X, k, init, num_restarts):
  best_obj = float("inf")
  best_C = None
  best_assignments = None
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(X.shape[0])
      C = X[perm[:k]]
    elif init == "kmeans++":
      C = kmeanspp_init(X, k)
    else:
      assert False
    C, assignments = lloyd_iteration(X, C)
    obj = kmeans_obj(X, C, assignments)
    if obj < best_obj:
      best_C = C.copy()
      best_assignments = assignments.copy()
      best_obj = obj
  return best_C, best_assignments, best_obj

def load_data():
  data = loadmat("Data/kmeans_data.mat")
  return data["X"]

if __name__ == "__main__":
  X = load_data()
  kmeans_cluster(X, 3, "random", 1)
