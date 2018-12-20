import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, identity, linalg
from sklearn.cluster import KMeans
import networkx as nx
import random

# Read file and create adjency, degree, identity, inverse degree and square root degree matrices
def read_graph():
    with open("./graphs_part_1/ca-CondMat.txt", "r") as lines:
        firstrow = True
        rows = []
        cols = []
        for idx, line in enumerate(lines):
            line = line.split()
            if firstrow:
                # graphID = line[1]
                vertices_number = int(line[2])
                edges_number = int(line[3])
                edges = np.empty((edges_number, 2))
                k = int(line[4])
                firstrow = False
                datad = [0] * vertices_number
            else:
                rows.append(int(line[0]))
                rows.append(int(line[1]))
                cols.append(int(line[0]))
                cols.append(int(line[1]))
                datad[int(line[0])] += 1
                datad[int(line[1])] += 1
                edges[idx-1] = [int(line[0]),int(line[1])]

    data = [1] * len(rows)
    A = csr_matrix((data, (rows, cols)))
    D = csr_matrix((datad, (list(range(vertices_number)), list(range(vertices_number)))))

    return A, D, edges, k, vertices_number, edges_number

def ng_norm(A, sqrt_D, I, k):
    L = I - sqrt_D * A * sqrt_D
    w, v = linalg.eigs(L, k)

    w = np.real(w)
    v = np.real(v)

    lengths = np.linalg.norm(v, axis=1)

    v = v / lengths[:, np.newaxis]

    U = v[:,:k] # first K eigenvectors (step 3)
    Y = v

    return Y, U

def shi_norm(A, inverse_D, D):
    L_normal = D - A
    L = np.matmul(inverse_D, L_normal) # Random walk normalized Laplacian
    w, v = np.linalg.eig(L) #, eigvals=(L.shape[0]-k, L.shape[0]-1)) #biggest eig
    U = np.real(v[:, :k]) # first K eigenvectors (step 3)
    return U

# cluster U into k clusters
def kmeans(Y, k):
    output = KMeans(n_clusters=k, n_jobs=-1).fit_predict(Y)
    fittrans = KMeans(n_clusters=k,random_state=0).fit_transform(Y)
    return output, fittrans

# objective function
def objective(r, k, edges):
    community_dict = {}
    size_coms = np.zeros((k))
    differentcommunity = 0

    for idx, i in enumerate(r):
        community_dict[idx] = i
        size_coms[i] += 1

    for edge in edges:
        unode, vnode = int(edge[0]), int(edge[1])
        ucom, vcom = community_dict[unode], community_dict[vnode]
        if ucom != vcom:
            differentcommunity += 1
    phi = differentcommunity/np.min(size_coms)
    return phi

def sizes(output, k):
    sizes = []

    for ka in range(k):
        k_array = []
        for idx, node in enumerate(output):
          if node == ka:
              k_array.append(idx)
        sizes.append(len(k_array))

    return sizes

def shuffle_communities(output, vertices_number, fittrans, k, sizes, edges):
    ITERACTION_COUNT = 5000
    min_phi = objective(output, k, edges)
    for i in range(ITERACTION_COUNT):
        o = output[:]
        a,b = random.sample(range(0, len(output) - 1), 2)
        tmp = o[a]
        o[a] = o[b]
        o[b] = tmp
        phi = objective(o, k, edges)
        if phi < min_phi:
            min_phi = phi
            output = o[:]

    return output, objective(output, k, edges)


def balance_communities(output, vertices_number, fittrans, k, sizes, edges):
    min_phi = objective(output, k, edges)
    changed = []
    for r in range(round(vertices_number / k)):
        for i in range(len(sizes)):
            # print(r, i)
            if sizes[i] > round(vertices_number / k):
                next_index = i + 1 if i < (len(sizes) - 1) else 0
                prev_index = i - 1 if i > 0 else len(sizes) - 1
                min_err = vertices_number
                movement = -1
                for j in range(len(output)):
                    if output[j] == i and j not in changed:
                        for ij in range(k):
                            if ij != i and fittrans[j][ij] < min_err:
                                min_err = fittrans[j][ij]
                                movement = ij
                                replace = j
                if movement != -1:
                    changed.append(replace)
                    output[replace] = movement
                    sizes[i] -= 1
                    sizes[movement] += 1
                phi = objective(output, k, edges)
                if phi < min_phi:
                    min_phi = phi
                    min_output = output[:]
                    cost = objective(output, k, edges)

    return min_output, cost

def main():
    A, D, edges, k, vertices_number, edges_number = read_graph()

    I = identity(vertices_number, format='csc')

    inverse_D = linalg.inv(D)
    sqrt_D = np.sqrt(inverse_D)

    Y, U = ng_norm(A, sqrt_D, I, k)
    print(Y)

    output, fittrans = kmeans(Y, k)

    s = sizes(output, k)

    print("Basic spectral algorithm")
    print(objective(output, k, edges))
    output, cost = balance_communities(output, vertices_number, fittrans, k, s, edges)
    with open('ca-CondMat-balanced.txt', 'a') as the_file:
        for idx, community in enumerate(output):
            the_file.write(str(idx)+" "+str(community)+"\n")
    print("Balanced result")
    print(cost)

main()


