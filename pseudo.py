# read the file and set the matrixes and variables
with open("./graphs_part_1/ca-GrQc.txt", "r") as lines:
  firstrow = True
  for line in lines:
    line = line.split()
    if firstrow:
      graphID = line[1]
      numOfVertices = int(line[2])
      numOfEdges = int(line[3])
      k = int(line[4]);
      A = [ [0] * numOfVertices for _ in range(numOfVertices)]
      D = [ [0] * numOfVertices for _ in range(numOfVertices)]
      firstrow = False
    else:
      A[int(line[0])][int(line[1])] = 1
      A[int(line[0])][int(line[1])] = 1
      D[int(line[0])][int(line[0])] += 1
      D[int(line[1])][int(line[1])] += 1

# TODO: compute matrix L
