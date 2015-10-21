import numpy
import math
import sys

_EPS = numpy.finfo(float).eps * 4.0

def get_frame(G, g):
    G = numpy.array(G)
    G_original = G
    g = numpy.array(g)
    g_original = g
    Gx, Gy, Gz = numpy.sum(G, axis=1)
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
    G = G - centroid_1 #center around origin
    # print(centroid_1)

    gx, gy, gz = numpy.sum(g, axis=1)
    centroid_2 = numpy.array([[gx], [gy], [gz]])/len(g[0])
    g = g - centroid_2 #center around origin

    xx, yy, zz = numpy.sum(G * g, axis=1)
    xy, yz, zx = numpy.sum(G * numpy.roll(g, -1, axis=0), axis=1)
    xz, yx, zy = numpy.sum(G * numpy.roll(g, -2, axis=0), axis=1)
    N = [[xx+yy+zz, yz-zy,      zx-xz,      xy-yx],
            [yz-zy,    xx-yy-zz, xy+yx,      zx+xz],
            [zx-xz,    xy+yx,    yy-xx-zz, yz+zy],
            [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
    w1, v1 = numpy.linalg.eig(N)
    max_index = numpy.argmax(w1)
    q = v1[:,max_index]
    n = numpy.dot(q, q)
    if n < _EPS:
        R = numpy.identity(3)
    else:
        rot_matrix = [[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]]
        R = numpy.array(rot_matrix)

    t = numpy.dot(R.T, centroid_2) - centroid_1

    return Frame(R.T, -t)


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



def B(k, v):
	# print(v)
	nCk = math.factorial(5) / (math.factorial(k)*math.factorial(5 - k))
	return nCk*((1 - v)**(5 - k))*(v**(k))

def get_coeff(C_expected, C):
	F = []
	C_norm = []
	# C_max = numpy.linalg.norm(numpy.amax(C, axis=0))
	# C_min = numpy.linalg.norm(numpy.amin(C, axis=0))
	C_max = []
	C_min = []
	C = numpy.array(C)
	for i in range(3):
		C_max.append(numpy.max(C[:,i]))
		C_min.append(numpy.min(C[:,i]))

	for i in range(len(C)):
		to_append = []
		for j in range(3):
			to_append.append((float(C[i][j]) - float(C_min[j]))/(float(C_max[j]) - float(C_min[j])))
		C_norm.append(to_append)
		# C_expected_norm.append((C_expected[i])/(C_max))

	# C_norm = C
	for data_point in range(len(C_norm)):
		F.append([])
		for i in range(6):
			for j in range(6):
				for k in range(6):
					F[data_point].append(B(i, C_norm[data_point][0])*B(j, C_norm[data_point][1])*B(k, C_norm[data_point][2]))

	return numpy.linalg.lstsq(numpy.array(F), numpy.array(C_expected))[0], C_min, C_max

def correct_distortion(coeffs, q, q_min, q_max):
	G_corrected = []
	q_norm = []
	# q_max = []
	# q_min = []
	# q = numpy.array(q)
	# for i in range(3):
	# 	q_max.append(numpy.max(q[:,i]))
	# 	q_min.append(numpy.min(q[:,i]))
	for i in range(len(q)):
		to_append = []
		# For each vector, take each component and normalize them
		for j in range(3):
			to_append.append((float(q[i][j]) - float(q_min[j]))/(float(q_max[j]) - float(q_min[j])))
		q_norm.append(to_append)


	F = []
	for data_point in range(len(q_norm)):
		# print(q_norm[data_point])
		# sum = [0, 0, 0]
		F.append([])
		for i in range(6):
			for j in range(6):
				for k in range(6):
					# co_index = 36*i + 6*j + k
					# print(sum)
					# sum = [sum[0] + coeffs[co_index][0]*B(i, q_norm[data_point][0]),  sum[1] + coeffs[co_index][1]*B(j, q_norm[data_point][1]), sum[2] + coeffs[co_index][2]*B(k, q_norm[data_point][2])]
					F[data_point].append(B(i, q_norm[data_point][0])*B(j, q_norm[data_point][1])*B(k, q_norm[data_point][2]))
					# print(coeffs[36*i + 6*j + k][0]*B(i, q_norm[data_point][0]), coeffs[36*i + 6*j + k][1]*B(j, q_norm[data_point][1]), coeffs[36*i + 6*j + k][2]*B(k, q_norm[data_point][2]))
		# print(sum)
		# G_corrected.append(sum)
	# print(coeffs.shape)
	F = numpy.array(F)
	# print(F.shape)
	G_corrected = numpy.dot(F, coeffs)
	# G_corrected = [G_corrected[0] - G_corrected[0], G_corrected[1] - G_corrected[0], G_corrected[2] - G_corrected[0], G_corrected[3] - G_corrected[0]]
	# G_corrected = numpy.array(G_corrected) - numpy.array(G_corrected[0])
	# print(G_corrected/G_corrected[1])
	return numpy.round(G_corrected, 3)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Main file
if (len(sys.argv) != 4):
	sys.exit(0)

cal_body = open(sys.argv[1])
cal_readings = open(sys.argv[2])
piv_points = open(sys.argv[3])
index1 = int(sys.argv[1].index("pa1"))
index2 = int(sys.argv[1].index("calbody"))
filename = sys.argv[1][index1:index2]

# cal_readings.readline()
first_line = cal_readings.readline().split(",")
N_d = int(first_line[0].strip())
N_a = int(first_line[1].strip())
N_c = int(first_line[2].strip())
N_frames = int(first_line[3].strip())
d = []
a = []
c = []
F_d = []
F_a = []
C = []
C_expected = []

piv_first_line = piv_points.readline().split(",")
num_markers = int(piv_first_line[0].strip())
G = []

# cal_body
for i in range(N_d):
	line = cal_body.readline().split(",")
	tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
	d.append(numpy.array(tpose).T)
for i in range(N_a):
	line = cal_body.readline().split(",")
	tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
	a.append(numpy.array(tpose).T)
for i in range(N_c):
	line = cal_body.readline().split(",")
	c.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])

# cal_readings
for i in range(N_frames):
	D = []
	A = []
	# C.append([])
	# C_expected.append([])
	# G.append([])
	for j in range(N_d):
		line = cal_readings.readline().split(",")
		tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		D.append(numpy.array(tpose).T)
	for j in range(N_a):
		line = cal_readings.readline().split(",")
		tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		A.append(numpy.array(tpose).T)
	F_d.append(get_frame(numpy.array(D).T, numpy.array(d).T))
	F_a.append(get_frame(numpy.array(A).T, numpy.array(a).T))
	R_d_i = numpy.linalg.inv(F_d[i].get_rot())
	P_d_i = numpy.dot(R_d_i, F_d[i].get_trans())
	P_a = F_a[i].get_trans()
	R_a = F_a[i].get_rot()
	for j in range(N_c):
		line = cal_readings.readline().split(",")
		tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		C.append(numpy.array(tpose).T)
		# C[i].append(numpy.array(tpose).T)
		inside = numpy.dot(R_a, c[j]) + P_a
		square = numpy.dot(R_d_i, inside) - P_d_i
		C_expected.append([square[0][0], square[1][1], square[2][2]])
		# C_expected[i].append([square[0][0], square[1][1], square[2][2]])
	

coeffs, q_min, q_max = get_coeff(C_expected, C)

G = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
print(correct_distortion(coeffs, G, q_min, q_max))

# frames = []
# rotations = []
# translations = []
# G0 = []
# g = []
# # print(numpy.allclose(correct_distortion(coeffs, C, q_min, q_max), C_expected, rtol=1e-02))
# for i in range(1):
# 	G = []
# 	for j in range(0, num_markers):
# 		line = piv_points.readline().split(",")
# 		# get array G1
# 		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
# 		G.append(t)
# 	G = numpy.array(G).T
# 	# G = numpy.array(G)
# 	G_corrected = correct_distortion(coeffs, G.T, q_min, q_max)
# 	print(G.T)
# 	print(G_corrected)
# 	# print(numpy.array(G).T)
# 	# print(G_corrected.T)
# 	# print(numpy.allclose(G, G_corrected, rtol=1e-01))

# 	if i is 0:
# 		Gx, Gy, Gz = numpy.sum(G_corrected, axis=1)
# 		G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G_corrected[0])
# 		g = G_corrected - G0
# 	frames.append(get_frame(G_corrected, g))
# 	curr_rot = numpy.array(frames[i].get_rot())
# 	rotations.append([curr_rot[0][0], curr_rot[0][1], curr_rot[0][2], -1, 0, 0])
# 	rotations.append([curr_rot[1][0], curr_rot[1][1], curr_rot[1][2], 0, -1, 0])
# 	rotations.append([curr_rot[2][0], curr_rot[2][1], curr_rot[2][2], 0, 0, -1])

# 	t = -1*frames[i].get_trans()
# 	translations.append(t[0])
# 	translations.append(t[1])
# 	translations.append(t[2])


# cal_body.close()
# cal_readings.close()
# piv_points.close()

# # # solve Pdimple = frames[k]*t
# a = numpy.squeeze(numpy.array(rotations))
# b = numpy.array(translations)
# x = numpy.linalg.lstsq(numpy.squeeze(numpy.array(rotations)), numpy.squeeze(numpy.array(translations)))
# print("%.2f" % x[0][3] + ", " + "%.2f" % x[0][4] + ", " + "%.2f" % x[0][5])





##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################



# if (len(sys.argv) != 2):
# 	sys.exit(0)

# in_file = open(sys.argv[1])

# frames = []
# rotations = []
# translations = []
# G0 = []
# g = []

# first_line = in_file.readline().split(",")
# num_markers = int(first_line[0].strip())
# num_frames = int(first_line[1].strip())
# for i in range(0, num_frames):
# 	G = []
# 	for j in range(0, num_markers):
# 		line = in_file.readline().split(",")
# 		# get array G1
# 		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
# 		G.append(t)
# 	G = numpy.array(G).T
# 	if i is 0:
# 		Gx, Gy, Gz = numpy.sum(G, axis=1)
# 		G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
# 		g = G - G0
# 	frames.append(get_frame(G, g))
# 	curr_rot = numpy.array(frames[i].get_rot())
# 	rotations.append([curr_rot[0][0], curr_rot[0][1], curr_rot[0][2], -1, 0, 0])
# 	rotations.append([curr_rot[1][0], curr_rot[1][1], curr_rot[1][2], 0, -1, 0])
# 	rotations.append([curr_rot[2][0], curr_rot[2][1], curr_rot[2][2], 0, 0, -1])

# 	t = -1*frames[i].get_trans()
# 	translations.append(t[0])
# 	translations.append(t[1])
# 	translations.append(t[2])


# # # solve Pdimple = frames[k]*t
# a = numpy.squeeze(numpy.array(rotations))
# b = numpy.array(translations)
# x = numpy.linalg.lstsq(numpy.squeeze(numpy.array(rotations)), numpy.squeeze(numpy.array(translations)))
# print("%.2f" % x[0][3] + ", " + "%.2f" % x[0][4] + ", " + "%.2f" % x[0][5])