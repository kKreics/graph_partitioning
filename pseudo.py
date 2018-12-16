import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
from random import random

# Graph similarity matrix
# Read file and create adjency, degree, identity, inverse degree and square root degree matrices
# ca-GrQc
with open("./graphs_part_1/karate.txt", "r") as lines:
  firstrow = True
  for idx, line in enumerate(lines):
    line = line.split()
    if firstrow:
      print("k: ", int(line[4]))
      graphID = line[1]
      vertices_number = int(line[2])
      edges_number = int(line[3])
      A = np.empty((vertices_number, vertices_number))
      D = np.empty((vertices_number, vertices_number))
      A = np.matrix(A)
      D = np.matrix(D)
      edges = np.empty((edges_number, 2))
      k = int(line[4])
      #A = [ [0] * vertices_number for _ in range(vertices_number)]
      #D = [ [0] * vertices_number for _ in range(vertices_number)]
      firstrow = False
    else:
      A[int(line[0]),int(line[1])] = 1
      A[int(line[1]),int(line[0])] = 1 # with these lines the whole adjency matrix is now calculated
      D[int(line[0]),int(line[0])] += 1
      D[int(line[1]),int(line[1])] += 1
      edges[idx-1] = [int(line[0]),int(line[1])]

np.savetxt("Normal_L.csv", D-A, delimiter=",", fmt='%f')
#print("A",A.shape)

print("A",A[:2,:2])
print("D",D[:2,:2])

np.savetxt("A.csv", A, delimiter=",", fmt='%f')
#np.savetxt("D.csv", D, delimiter=",", fmt='%f')
I = np.identity(vertices_number)

#np.savetxt("I.csv", I, delimiter=",", fmt='%f')
try:
    inverse_D = np.linalg.inv(D)
    #print(inverse_D)
except:
    print("cant inverse")
sqrt_D = sp.linalg.sqrtm(inverse_D)
#np.savetxt("Inverse_D.csv", inverse_D, delimiter=",", fmt='%f')
def ng_norm(A,sqrt_D):
    L = np.subtract(I, np.matmul(sqrt_D, np.matmul(A, sqrt_D))) # Symmetric normalized Laplacian
    np.savetxt("Symmetric_Normalized_L.csv", L, delimiter=",", fmt='%f')
    w, v = np.linalg.eig(L) #, eigvals=(L.shape[0]-k, L.shape[0]-1)) #biggest eig
    #print("k eignevalues:",k,w)
    U = v[:,:k] # first K eigenvectors (step 3)
    Y = np.zeros(U.shape)
    # print("L shape",L.shape)
    # print("Y shape",Y.shape)
    # print("U shape",U.shape)
    # normalize rows (step 4)
    for i in range(U.shape[0]):
        sum = np.sum(np.square(U[i]))
        denominator = np.sqrt(sum)
        for j in range(U.shape[1]):
            Y[i,j] = U[i,j]/denominator # Step 4

    #np.savetxt("U.csv", U, delimiter=",", fmt='%f')
    #np.savetxt("Y.csv", Y, delimiter=",", fmt='%f')
    #np.savetxt("v.csv", v, delimiter=",", fmt='%f')
    return Y, U

def shi_norm(A, inverse_D, D):
    L_normal = D - A
    L = np.matmul(inverse_D, L_normal) # Random walk normalized Laplacian
    #np.savetxt("Random_Walk_Normalized_L.csv", L, delimiter=",", fmt='%f')
    #print("L",L[:50,:50])
    w, v = np.linalg.eig(L) #, eigvals=(L.shape[0]-k, L.shape[0]-1)) #biggest eig
    #print("k eignevalues:", k, w)
    U = np.real(v[:, :k]) # first K eigenvectors (step 3)
    #print("L shape", L.shape)
    #print("Y shape",Y.shape)
    #print("U shape",U.shape)
    # normalize rows (step 4)
    #np.savetxt("U.csv", U, delimiter=",")
    #np.savetxt("Y.csv", Y, delimiter=",")
    #np.savetxt("v.csv", v, delimiter=",")
    return U

#Cluster U into k clusters
def kmeans(Y):
    kmeans = KMeans(n_clusters=k, random_state=0,max_iter=1000).fit(Y)
    fittrans = KMeans(n_clusters=k,random_state=0).fit_transform(Y)
    output = kmeans.predict(Y)
    return output, fittrans


#Y = shi_norm(A, inverse_D, D)
Y, U = ng_norm(A, sqrt_D)
output, fittrans = kmeans(Y)

community_dict = {}
cutted_edges = np.zeros((k))
size_coms = np.zeros((k))
samecommunity = 0
differentcommunity = 0

for idx, i in enumerate(output):
    community_dict[idx] = i
    size_coms[i] += 1
#print(community_dict)
print(output)
print(fittrans)


# minmax_kmeans
#np.savetxt('data.data', Y, delimiter='   ')
#os.system("./minmax_kmeans.py ../data.data 3 2 100 -n 3 -o mm.out")




# objective function
for edge in edges:
    #print(edge)
    unode, vnode = int(edge[0]), int(edge[1])
    ucom, vcom = community_dict[unode], community_dict[vnode]
    if ucom == vcom:
        samecommunity += 1
    else:
        differentcommunity += 1
        cutted_edges[ucom] += 1
        cutted_edges[vcom] += 1
#print("Total non-cutted edges:",samecommunity)
#print("Total cutted edges:",differentcommunity)

for com in range(k):
    print("Community",com,"cuts",int(cutted_edges[com]),"edges")

#print(cutted_edges)
#print(size_coms)
#print("Size of smallest community:",int(np.min(size_coms)))




#OBJECTIVE aka Isoperimetric number (https://arxiv.org/pdf/1609.08072.pdf, p. 12)
phi = cutted_edges/np.min(size_coms)
print("Objective:",phi)

def plot():
    comsize = []
    nodes_dict = {}
    for ka in range(k):
        k_array = []
        for idx, node in enumerate(output):
            if node == ka:
                k_array.append(idx)
        nodes_dict[ka] = k_array
        print("Community",int(ka),"has",len(k_array),"nodes.")

        comsize.append(len(k_array))

    print(comsize)



    print("nodes_dict:",nodes_dict)
    colors = [(random(), random(), random()) for _i in range(k)]
    tupleedges = tuple(map(tuple, edges.astype(int)))
    G=nx.Graph()
    for com in range(k):
        G.add_nodes_from(nodes_dict[com])
    G.add_edges_from(tupleedges)

    pos = nx.spring_layout(G)
    for com in range(k):
        nx.draw(G,pos=pos, node_size=150,nodelist = nodes_dict[com], node_color=colors[com])



    print("Node Degree")
    degree = []
    for v in G:
        degree.append(G.degree(v))
    #print("Average degree:",np.average(degree))
    #print("Std. degree:",np.std(degree))
    #plt.boxplot(degree)
    plt.show()
    return




plot()
