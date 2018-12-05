import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

# read the file and set the matrixes and variables
with open("./graphs_part_1/ca-GrQc.txt", "r") as lines:
  firstrow = True
  for line in lines:
    line = line.split()
    if firstrow:
      print("k: ", int(line[4]))
      graphID = line[1]
      vertices_number = int(line[2])
      edges_number = int(line[3])
      k = int(line[4]);
      A = [ [0] * vertices_number for _ in range(vertices_number)]
      D = [ [0] * vertices_number for _ in range(vertices_number)]
      firstrow = False
    else:
      A[int(line[0])][int(line[1])] = 1
      A[int(line[0])][int(line[1])] = 1
      D[int(line[0])][int(line[0])] += 1
      D[int(line[1])][int(line[1])] += 1

I = np.identity(vertices_number)
inverse_D = np.linalg.inv(D)
sqrt_D = sp.linalg.sqrtm(inverse_D)
L = np.subtract(I, np.matmul(sqrt_D, np.matmul(A, sqrt_D)))
w, v = np.linalg.eigh(L)

# print first k eigenvectors
#print(v)

print(v[:k].shape)
U = np.matrix(v[:k])
# TODO: create U, normalize U, cluster points and output clusters
