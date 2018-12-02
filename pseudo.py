with open("./graphs_part_1/ca-GrQc.txt", "r") as lines:
  firstrow = True;
  for line in lines:
    line = line.split();
    if firstrow:
      graphID = line[1];
      numOfVertices = int(line[2]);
      numOfEdges = int(line[3]);
      k = int(line[4]);
      A = [ [0] * numOfVertices for _ in range(numOfVertices)]
      D = [ [0] * numOfVertices for _ in range(numOfVertices)]
      firstrow = False;
    # else:

# print(A, D);
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
