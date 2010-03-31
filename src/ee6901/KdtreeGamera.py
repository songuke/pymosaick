from gamera.kdtree import *

points = [(1,4), (2,4), (1,5), (3,6), (8,9),
          (2,7), (4,4), (5,5), (4,6), (8,3)]
nodes = [KdNode(p) for p in points]
tree = KdTree(nodes)

# neighbors to a sample point not from the set
point = [5,6]
k = 3
knn = tree.k_nearest_neighbors(point, k)
print "%i neighbors of (%i,%i):" % (k,point[0], point[1]),
for node in knn:
    print "(%i,%i)" % (node.point[0], node.point[1]),
print "" # final newline

# neighbors to a sample point from the set
# we must query k+1 neighbors, because one of them is
# the sample point (the first entry in the returned list)
point = [5,5]
k = 3
knn = tree.k_nearest_neighbors(point, k+1)
print "%i neighbors of (%i,%i):" % (k,point[0], point[1]),
for node in knn[1:]:
    print "(%i,%i)" % (node.point[0], node.point[1]),
print "" # final newline
