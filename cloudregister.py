import numpy
import math
from transformations import superimposition_matrix, random_rotation_matrix

R = random_rotation_matrix(numpy.random.random(3))
v0 = [[1,3,0], [4,1,0], [8,0,1], [0,0,1]]
print (R)
v1 = numpy.dot(R, v0)
M = superimposition_matrix(v0, v1)
print (numpy.allclose(v1, numpy.dot(M, v0)))