import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import networkx as nx
import random

# Read file and create adjency, degree, identity, inverse degree and square root degree matrices
def read_graph():
    with open("./graphs_part_1/ca-GrQc.txt", "r") as lines:
        firstrow = True
        for idx, line in enumerate(lines):
            line = line.split()
            if firstrow:
                # graphID = line[1]
                vertices_number = int(line[2])
                edges_number = int(line[3])
                A = np.empty((vertices_number, vertices_number))
                D = np.empty((vertices_number, vertices_number))
                A = np.matrix(A)
                D = np.matrix(D)
                edges = np.empty((edges_number, 2))
                k = int(line[4])
                firstrow = False
            else:
                A[int(line[0]),int(line[1])] = 1
                A[int(line[1]),int(line[0])] = 1 # with these lines the whole adjency matrix is now calculated
                D[int(line[0]),int(line[0])] += 1
                D[int(line[1]),int(line[1])] += 1
                edges[idx-1] = [int(line[0]),int(line[1])]

    return A, D, edges, k, vertices_number, edges_number

def ng_norm(A, sqrt_D, I, k):
    L = np.subtract(I, np.matmul(sqrt_D, np.matmul(A, sqrt_D))) # Symmetric normalized Laplacian
    w, v = np.linalg.eig(L) #, eigvals=(L.shape[0]-k, L.shape[0]-1)) #biggest eig
    U = v[:,:k] # first K eigenvectors (step 3)
    Y = np.zeros(U.shape)
    for i in range(U.shape[0]):
        sum = np.sum(np.square(U[i]))
        denominator = np.sqrt(sum)
        for j in range(U.shape[1]):
            Y[i,j] = U[i,j]/denominator # Step 4

    return Y, U

def shi_norm(A, inverse_D, D):
    L_normal = D - A
    L = np.matmul(inverse_D, L_normal) # Random walk normalized Laplacian
    w, v = np.linalg.eig(L) #, eigvals=(L.shape[0]-k, L.shape[0]-1)) #biggest eig
    U = np.real(v[:, :k]) # first K eigenvectors (step 3)
    return U

# cluster U into k clusters
def kmeans(Y, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Y)
    fittrans = KMeans(n_clusters=k,random_state=0).fit_transform(Y)
    output = kmeans.predict(Y)
    return output, fittrans

# objective function
def objective(r, k, edges):
    community_dict = {}
    cutted_edges = np.zeros((k))
    size_coms = np.zeros((k))
    samecommunity = 0
    differentcommunity = 0

    for idx, i in enumerate(r):
        community_dict[idx] = i
        size_coms[i] += 1

    for edge in edges:
        unode, vnode = int(edge[0]), int(edge[1])
        ucom, vcom = community_dict[unode], community_dict[vnode]
        if ucom == vcom:
            samecommunity += 1
        else:
            differentcommunity += 1
            cutted_edges[ucom] += 1
            cutted_edges[vcom] += 1
    phi = cutted_edges/np.min(size_coms)
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
    min_phi = vertices_number
    for i in range(ITERACTION_COUNT):
        o = output[:]
        a,b = random.sample(range(0, len(output) - 1), 2)
        tmp = o[a]
        o[a] = o[b]
        o[b] = tmp
        phi = sum(objective(o, k, edges))
        if phi < min_phi:
            min_phi = phi
            output = o[:]

    return output, objective(output, k, edges)


def balance_communities(output, vertices_number, fittrans, k, sizes, edges):
    min_phi = vertices_number
    for r in range(round(vertices_number / k)):
        for i in range(len(sizes)):
            if sizes[i] > round(vertices_number / k):
                next_index = i + 1 if i < (len(sizes) - 1) else 0
                prev_index = i - 1 if i > 0 else len(sizes) - 1
                min_err = vertices_number
                movement = -1
                for j in range(len(output)):
                    if output[j] == i:
                        if fittrans[j][next_index] < min_err or fittrans[j][prev_index] < min_err:
                            min_err = fittrans[j][prev_index] if fittrans[j][next_index] > fittrans[j][prev_index] else fittrans[j][next_index]
                            movement = prev_index if fittrans[j][next_index] > fittrans[j][prev_index] else next_index
                            replace = j
                if movement != -1:
                    output[replace] = movement
                    sizes[i] -= 1
                    sizes[movement] += 1
                phi = sum(objective(output, k, edges))
                if phi < min_phi:
                    min_phi = phi
                    min_output = output[:]
                    objective = objective(output, k, edges)

    return min_output, objective

def main():
    A, D, edges, k, vertices_number, edges_number = read_graph()

    I = np.identity(vertices_number)

    inverse_D = np.linalg.inv(D)
    sqrt_D = sp.linalg.sqrtm(inverse_D)

    Y, U = ng_norm(A, sqrt_D, I, k)
    output, fittrans = kmeans(Y, k)

    s = sizes(output, k)
    # output, objective = balance_communities(output, vertices_number, fittrans, k, s, edges)
    output, objective = shuffle_communities(output, vertices_number, fittrans, k, s, edges)

    print("Result")
    print(objective)
    print(output)

main()


