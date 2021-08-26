import numpy as np
import matplotlib.pyplot as plt
import kmeans

X = kmeans.load_data()

k = np.arange(1,21)
get_obj = lambda i: kmeans.kmeans_cluster(X, i, 'random', 10)[2]
obj = [get_obj(i) for i in k]
plt.plot(k, obj)
plt.xlabel("k")
plt.ylabel("k-means objective")
plt.savefig('kmeans_objective.png')
plt.show()

init = ['random', 'kmeans++']
N = 1000
obj_rand = np.mean([kmeans.kmeans_cluster(X, 9, 'random', 1)[2] for i in range(N)])
obj_plusplus = np.mean([kmeans.kmeans_cluster(X, 9, 'kmeans++', 1)[2] for i in range(N)])
print("random: {}\nkmeans++: {}".format(obj_rand, obj_plusplus))