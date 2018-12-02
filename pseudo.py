# open text file
# read the first line (# graphID numOfVertices numOfEdges k)
# save k, numOfEdges and graphID
# create an empty (all 0) numOfVerticesxnumOfVertices adjancency matrix A
# create an empty (all 0) numOfVerticesxnumOfVertices diagonal matrix D
# read edges (vertex1ID vertex2ID) line by line until you reach the numOfEdges line
#   on each read
#     put 1 in A[vertex1ID][vertex2ID]
#     put 1 in A[vertex2ID][vertex1ID]
#     increment D[vertex1ID][vertex1ID] by 1
#     increment D[vertex2ID][vertex2ID] by 1
# compute matrix L
