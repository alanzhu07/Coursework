import numpy as np
import submission as sub
import matplotlib.pyplot as plt

data = np.load("../data/pnp.npz", allow_pickle=True)
x, X = data['x'], data['X']
image = data['image']
cad = data['cad'][0][0][0]

P = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(P)

X_h = np.insert(X, 3, 1, axis=1)
projected = X_h @ P.T
projected = projected[:,:2] / projected[:,[2]]

plt.imshow(image)
plt.scatter(x[:,0], x[:,1], edgecolors='g', s=20, facecolors='none', label='original')
plt.scatter(projected[:,0], projected[:,1], c='m', s=2, label='projected')
plt.legend()
plt.show()

cad_rotated = cad @ R

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(cad_rotated[:,0], cad_rotated[:,1], cad_rotated[:,2], '-o', c='b', linewidth=0.3, markersize=0.1, alpha=0.6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

cad_projected = np.insert(cad, 3, 1, axis=1) @ P.T
cad_projected = cad_projected[:,:2] / cad_projected[:,[2]]

plt.imshow(image)
plt.plot(cad_projected[:,0], cad_projected[:,1], '-o', c='r', linewidth=0.3, markersize=0.4, alpha=0.6)
plt.show()