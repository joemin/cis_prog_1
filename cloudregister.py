import numpy
import math
import sys
from transformations import superimposition_matrix, random_rotation_matrix, quaternion_matrix

"""def get_registration(v0, v1):
	v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
	v1 = numpy.array(v1, dtype=numpy.float64, copy=True)
	# print(v0)
	# print(v1)
	# print(v0*v1)
	xx, yy, zz = numpy.sum(v0 * v1, axis=0)
	xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=1), axis=0)
	xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=1), axis=0)
	H = [[xx, xy, xz],
	[yx, yy, yz],
	[zx, zy, zz]]
	# print(numpy.array(H).T)
	trace_n = (xx + yy + zz)

	# print(trace_n)

	delta = [yz - zy, zx - xz, xy - yx]
	# delta_t = numpy.array(delta).T
	# print(H + numpy.array(H).T)
	# print("---------------------------------------")
	# print(numpy.array(H) + numpy.array(H).T)
	mat_3 = numpy.array(H) + numpy.array(H).T - trace_n
	# print(mat_3)
	# print("---------------------------------------")
	# G = [[trace_n, numpy.array(delta).T], [delta, H + numpy.array(H).T - trace_n]]
	G = [[trace_n, delta[0], delta[1], delta[2]],
	[delta[0], mat_3[0][0], mat_3[0][1], mat_3[0][2]],
	[delta[1], mat_3[1][0], mat_3[1][1], mat_3[1][2]],
	[delta[2], mat_3[2][0], mat_3[2][1], mat_3[2][2]]]

	# print(numpy.array(G))

	w1, v1 = numpy.linalg.eig(G)
	max_index = numpy.argmax(w1)
	w, V = numpy.linalg.eigh(G)
	q_new = V[:, numpy.argmax(w)]
	q = v1[:,max_index]
	# print("0000000000000")
	# print(q_new)
	# print(q)
	# print("00000000000000")
	#q /= vector_norm(q)  # unit quaternion
	# homogeneous transformation matrix
	M = quaternion_matrix(q)

	# print(M)

	# q = v[max_index]
	# print(q)
	# q = [0, 0, 0, 0]

	# print(numpy.array(v0).T * numpy.array(rot_matrix).T)

	rot_matrix = [[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
	[2*(q[1]*q[2] + q[0]*q[3]), math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[2]*q[3] - q[0]*q[1])],
	[2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]]

	#rot_matrix = [[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
	#[2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
	#[2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]]
	#print(numpy.array(rot_matrix))

	R = numpy.array(rot_matrix)

	# print(R)
	# return R
	Gx, Gy, Gz = numpy.sum(v0, axis=0)
	centroid_1 = numpy.dot(R, numpy.array([Gx, Gy, Gz])/len(v0))

	gx, gy, gz = numpy.sum(v1, axis=0)
	centroid_2 = numpy.array([gx, gy, gz])/len(v1)

	t = centroid_1 - centroid_2
	return Frame(R, t)"""

def get_frame(G, g):
	G = numpy.array(G)
	g = numpy.array(g)
	xx, yy, zz = numpy.sum(G * g, axis=0)
	xy, yz, zx = numpy.sum(G * numpy.roll(g, -1, axis=1), axis=0)
	xz, yx, zy = numpy.sum(G * numpy.roll(g, -2, axis=1), axis=0)
	H = [[xx, xy, xz],
	[yx, yy, yz],
	[zx, zy, zz]]
	trace_n = (xx + yy + zz)
	delta = [yz - zy, zx - xz, xy - yx]
	mat_3 = numpy.array(H) + numpy.array(H).T - trace_n
	mat_4 = [[trace_n, delta[0], delta[1], delta[2]],
	[delta[0], mat_3[0][0], mat_3[0][1], mat_3[0][2]],
	[delta[1], mat_3[1][0], mat_3[1][1], mat_3[1][2]],
	[delta[2], mat_3[2][0], mat_3[2][1], mat_3[2][2]]]
	w1, v1 = numpy.linalg.eig(mat_4)
	max_index = numpy.argmax(w1)
	q = v1[:,max_index]
	rot_matrix = [[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
	[2*(q[1]*q[2] + q[0]*q[3]), math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[2]*q[3] - q[0]*q[1])],
	[2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]]
	R = numpy.array(rot_matrix)

	# print(G)
	Gx, Gy, Gz = numpy.sum(G, axis=0)
	# print(Gx, Gy, Gz)
	# print(len(G))
	centroid_1 = numpy.dot(R, numpy.array([Gx, Gy, Gz])/len(G))
	# print(centroid_1)

	gx, gy, gz = numpy.sum(g, axis=0)
	centroid_2 = numpy.array([gx, gy, gz])/len(g)
	# print(centroid_2)

	t = numpy.array(centroid_1 - centroid_2)
	# print(t)
	return Frame(R, t)


class Frame:

	def __init__(self, rotation = None, translation = None):
		if rotation is None:
			self.rotation = [numpy.identity(3)] #the identity matrix should be default
		else:
			self.rotation = rotation
		if translation is None:
			self.translation = numpy.array([0, 0, 0]) #default translation should be zero
		else:
			self.translation = translation

	def set_rot(self, rotation):
		self.rotation = rotation

	def get_rot(self):
		return self.rotation

	def set_trans(self, translation):
		self.translation = translation

	def get_trans(self):
		return self.translation


if (len(sys.argv) != 3):
	sys.exit(0)

cal_body = open(sys.argv[1])
cal_readings = open(sys.argv[2])

cal_body.readline()
first_line = cal_readings.readline().split(",")
N_d = int(first_line[0].strip())
N_a = int(first_line[1].strip())
N_c = int(first_line[2].strip())
N_frames = int(first_line[3].strip())
d = []
a = []
c = []
frames = []
F_d = []
F_a = []
C_expected = []

# cal_body
for i in range(N_d):
	line = cal_body.readline().split(",")
	d.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
for i in range(N_a):
	line = cal_body.readline().split(",")
	a.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
for i in range(N_c):
	line = cal_body.readline().split(",")
	c.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])

# cal_readings
for i in range(N_frames):
	D = []
	A = []
	C = []
	for j in range(N_d):
		line = cal_readings.readline().split(",")
		D.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
	for j in range(N_a):
		line = cal_readings.readline().split(",")
		A.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
	F_d.append(get_frame(D, d))
	F_a.append(get_frame(A, a))
	
	R_d_i = numpy.linalg.inv(F_d[i].get_rot())
	P_d_i = numpy.dot(R_d_i, F_d[i].get_trans())
	P_a = F_a[i].get_trans()
	R_a = F_a[i].get_rot()
	# print(new_trans)
	# for each frame, calculate Fd and Fa, and compute C expected using Fd-1 * Fa * c
	for j in range(N_c):
		line = cal_readings.readline().split(",") # this is unnecessary
		# C.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		# C_expected.append(numpy.array(numpy.dot(R_d_i_R_a, c[j]) + new_trans))
		inside = numpy.dot(R_a, c[j]) + P_a
		C_expected.append(numpy.dot(R_d_i, inside) - P_d_i)
		# C_expected.append(numpy.dot(numpy.dot(numpy.linalg.inv(numpy.array(F_d[i])), numpy.array(F_a[i])), numpy.array(c[j])))
		print(C_expected[j])
		print(line)
		print("***")
	print("=============================================", i)

# Change this later to be in the loop above
# for i in range(N_frames):



# R = random_rotation_matrix(numpy.random.random(3))
# v0 = [[0,0,0], [1,2,3], [2,3,1], [3,2,1]]
# v1 = [[2,3,0], [8,1,2], [8,3,4], [2,0,1]]
# v1 = [[0,0,0], [1,2,3], [2,3,1], [3,2,1]]

#v1 = numpy.dot(R, v0)





# print(v0*rot_matrix)

# R = random_rotation_matrix(numpy.random.random(3))
# v0 = [[1,3,0], [4,1,0], [8,0,1], [0,0,1]]
# print (R)
# v1 = numpy.dot(R, v0)
# print(v1)
# M = superimposition_matrix(v0, v1)
# print(numpy.dot(M, v0))
# print (numpy.allclose(v1, numpy.dot(M, v0)))