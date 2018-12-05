import numpy as np
import scipy as sp
from sklearn.cluster import KMeans



# read the file and set the matrixes and variables
with open("./graphs_part_1/karate.txt", "r") as lines:
  firstrow = True
  for idx, line in enumerate(lines):
    line = line.split()
    if firstrow:
      print("k: ", int(line[4]))
      graphID = line[1]
      vertices_number = int(line[2])
      edges_number = int(line[3])
      edges = np.empty((edges_number, 2))
      k = int(line[4]);
      A = [ [0] * vertices_number for _ in range(vertices_number)]
      D = [ [0] * vertices_number for _ in range(vertices_number)]
      firstrow = False
    else:
      A[int(line[0])][int(line[1])] = 1
      A[int(line[0])][int(line[1])] = 1
      A[int(line[1])][int(line[0])] = 1 # with these lines the whole adjency matrix is now calculated
      A[int(line[1])][int(line[0])] = 1
      D[int(line[0])][int(line[0])] += 1
      D[int(line[1])][int(line[1])] += 1
      edges[idx-1] = [int(line[0]),int(line[1])]
np.savetxt("A.csv", A, delimiter=",")
np.savetxt("D.csv", D, delimiter=",")
I = np.identity(vertices_number)
np.savetxt("I.csv", I, delimiter=",")
inverse_D = np.linalg.inv(D)
sqrt_D = sp.linalg.sqrtm(inverse_D)
L = np.subtract(I, np.matmul(sqrt_D, np.matmul(A, sqrt_D)))
np.savetxt("Normalized_L.csv", L, delimiter=",")
w, v = sp.linalg.eigh(L, eigvals=(L.shape[0]-k, L.shape[0]-1))

# print first k eigenvectors
#print("eigenvectors", v.shape)
#print(v[:k].shape)
U = v # K largest eigenvectors (step 3)
Y = np.zeros(U.shape)
#print("y shape",Y.shape)
#normalize rows (step 4)
for i in range(U.shape[0]):
    #print("i", i)
    #print(U[i])
    sum = np.sum(np.square(U[i]))
    #print("sum: ",sum)
    denominator = np.sqrt(sum)
    #print("denominator: ", denominator)
    for j in range(U.shape[1]):
        #print("j", j)
        Y[i][j] = U[i][j]/denominator # Step 4
# TODO: create U, normalize U, cluster points and output clusters

#Cluster U into k clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
output = kmeans.predict(U)
#print(U[:5,:])
np.savetxt("U.csv", U, delimiter=",")
#print(Y[:5,:])
np.savetxt("Y.csv", Y, delimiter=",")
print(output)
np.savetxt("v.csv", v, delimiter=",")

community_dict = {}
for idx, i in enumerate(output):
    community_dict[idx] = i
#print(community_dict)

cutted_edges = np.zeros((k))
# objective function
samecommunity = 0
differentcommunity = 0
for edge in edges:
    print(edge)
    unode,vnode = int(edge[0]),int(edge[1])
    ucom,vcom = community_dict[unode],community_dict[vnode]
    if ucom == vcom:
        samecommunity += 1
    else:
        differentcommunity += 1
        cutted_edges[ucom] += 1
        cutted_edges[vcom] += 1
print("Total non-cutted edges:",samecommunity)
print("Total cutted edges:",differentcommunity)
for com in range(k):
    print("Community",com,"cuts",cutted_edges[com],"edges")
#print(cutted_edges)
