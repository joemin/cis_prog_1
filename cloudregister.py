import numpy
import math
import sys
from transformations import superimposition_matrix, random_rotation_matrix, quaternion_matrix

# get v0, v1

if (len(sys.argv) != 3):
	sys.exit(0)

cal_body = open(sys.argv[1])
cal_readings = open(sys.argv[2])

cal_body.readline()
first_line = cal_readings.readline().split(",")
N_d = first_line[0].strip()
N_a = first_line[1].strip()
N_c = first_line[2].strip()
N_frames = first_line[3].strip()
d = []
a = []
c = []
frames = []

# cal_body
for i in range(N_d):
	line = cal_body.readline().split()
	d.append([line[0].strip(), line[1].strip(), line[3].strip()])
for i in range(N_a):
	line = cal_body.readline().split()
	a.append([line[0].strip(), line[1].strip(), line[3].strip()])
for i in range(N_c):
	line = cal_body.readline().split()
	c.append([line[0].strip(), line[1].strip(), line[3].strip()])

# cal_readings
for i in range(N_frames):
	D = []
	A = []
	C = []
	for j in range(N_d):
		D.append([line[0].strip(), line[1].strip(), line[3].strip()])
	for j in range(N_a):
		line = cal_body.readline().split()
		A.append([line[0].strip(), line[1].strip(), line[3].strip()])
	for j in range(N_c):
		line = cal_body.readline().split()
		C.append([line[0].strip(), line[1].strip(), line[3].strip()])
	frames.append[D, A, C]

# for each frame, calculate Fd and Fa, and compute C expected using Fd-1 * Fa * c

R = random_rotation_matrix(numpy.random.random(3))
v0 = [[0,0,0], [1,2,3], [2,3,1], [3,2,1]]
# v1 = [[2,3,0], [8,1,2], [8,3,4], [2,0,1]]
v1 = [[0,0,0], [1,2,3], [2,3,1], [3,2,1]]

#v1 = numpy.dot(R, v0)

v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

#============================================
print(v0)
print(v1)
# print(v0*v1)
xx, yy, zz = numpy.sum(v0 * v1, axis=0)
xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=1), axis=0)
xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=1), axis=0)
H = [[xx, xy, xz],
[yx, yy, yz],
[zx, zy, zz]]
print(numpy.array(H).T)
trace_n = (xx + yy + zz)

print(trace_n)

delta = [yz - zy, zx - xz, xy - yx]
# delta_t = numpy.array(delta).T
# print(H + numpy.array(H).T)
print("---------------------------------------")
print(numpy.array(H) + numpy.array(H).T)
mat_3 = numpy.array(H) + numpy.array(H).T - trace_n
print(mat_3)
print("---------------------------------------")
# G = [[trace_n, numpy.array(delta).T], [delta, H + numpy.array(H).T - trace_n]]
G = [[trace_n, delta[0], delta[1], delta[2]],
[delta[0], mat_3[0][0], mat_3[0][1], mat_3[0][2]],
[delta[1], mat_3[1][0], mat_3[1][1], mat_3[1][2]],
[delta[2], mat_3[2][0], mat_3[2][1], mat_3[2][2]]]

print(numpy.array(G))

w1, v1 = numpy.linalg.eig(G)
max_index = numpy.argmax(w1)
w, V = numpy.linalg.eigh(G)
q_new = V[:, numpy.argmax(w)]
q = v1[:,max_index]
print("0000000000000")
print(q_new)
print(q)
print("00000000000000")
#q /= vector_norm(q)  # unit quaternion
# homogeneous transformation matrix
M = quaternion_matrix(q)

# print(M)

"""q = v[max_index]
print(q)
q = [0, 0, 0, 0]"""

# print(numpy.array(v0).T * numpy.array(rot_matrix).T)

rot_matrix = [[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
[2*(q[1]*q[2] + q[0]*q[3]), math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[2]*q[3] - q[0]*q[1])],
[2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]]

#rot_matrix = [[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
#[2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
#[2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]]
#print(numpy.array(rot_matrix))

R = numpy.array(rot_matrix)

print(R)
# print(v0*rot_matrix)

# R = random_rotation_matrix(numpy.random.random(3))
# v0 = [[1,3,0], [4,1,0], [8,0,1], [0,0,1]]
# print (R)
# v1 = numpy.dot(R, v0)
# print(v1)
# M = superimposition_matrix(v0, v1)
# print(numpy.dot(M, v0))
# print (numpy.allclose(v1, numpy.dot(M, v0)))