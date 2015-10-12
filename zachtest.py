import numpy
import math
import sys
#from transformations import superimposition_matrix, random_rotation_matrix, quaternion_matrix

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
"""
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
"""

d = [[0.00,     0.00, -1500.00], [0.00,     0.00, -1350.00], [0.00,   150.00, -1500.00]]

D = [[0.00,     0.00,     0.00], [0.00,     0.00,   150.00],  [0.00,   150.00,     0.00]]

f = get_frame(D,d)
print("test")
print(f.get_rot())
print(f.get_trans())