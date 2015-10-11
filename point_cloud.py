import numpy
import math
from transformations import affine_matrix_from_points, translation_matrix, random_rotation_matrix, scale_matrix, concatenate_matrices, superimposition_matrix

# def transform(matrix1, matrix2):

# v0 = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
# v1 = [[0, 0, 0], [-1, -1, -1], [-2, -2, -2]]
# mat = superimposition_matrix(v0, v1)
# print(mat)
# v3 = numpy.dot(numpy.array(mat).T, numpy.array(v0))
# print(v3)
# newV = numpy.dot(mat, numpy.array(v1))
# print(newV)


# v0 = numpy.random.rand(3, 3)
# print(v0)
# M = superimposition_matrix(v0, v1)
# print(numpy.allclose(numpy.dot(v0, M), v1))
R = random_rotation_matrix(numpy.random.random(3))
# print(R)
v0 = [[1,0,0], [0,1,0], [0,0,1]]
v1 = numpy.dot(R, v0)
M = superimposition_matrix(v0, v1)
print(numpy.allclose(v1, numpy.dot(M, v0)))