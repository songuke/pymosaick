import numpy as np
from scipy.spatial import KDTree
x, y = np.mgrid[0:5, 2:8]
tree = KDTree(zip(x.ravel(), y.ravel()))
print tree.data
pts = np.array([[0, 0], [2.1, 2.9]])
tree.query(pts)
