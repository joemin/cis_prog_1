import numpy
import math
import sys
import random

_EPS = numpy.finfo(float).eps * 4.0

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Returns F such that F*g = G
def get_frame(G, g):
    G = numpy.array(G)
    G_original = G
    g = numpy.array(g)
    g_original = g
    Gx, Gy, Gz = numpy.sum(G, axis=1)
    # print(Gx, Gy, Gz)
    # print(Gx, Gy, Gz)
    # print(len(G))
    # print(len(G[0]))
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
    G = G - centroid_1 #center around origin
    # print(centroid_1)

    gx, gy, gz = numpy.sum(g, axis=1)
    # print(gx, gy, gz)
    centroid_2 = numpy.array([[gx], [gy], [gz]])/len(g[0])
    g = g - centroid_2 #center around origin
    # print(g)
    # print()
    # print(centroid_2)

    # t = numpy.array(centroid_2 - centroid_1)
    # G = G + t

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
    #print (_EPS)
    if n < _EPS:
        R = numpy.identity(3)
    else:
        rot_matrix = [[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]]
        R = numpy.array(rot_matrix)

    t = numpy.dot(R.T, centroid_2) - centroid_1
    # print(t)
    # t = numpy.zeros((3,1))
    # if (t[0][0] < .00000000001):
    #     t[0][0] = 0.00
    # if (t[1][0] < .00000000001):
    #     t[1][0] = 0.00
    # if (t[2][0] < .00000000001):
    #     t[2][0] = 0.00

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

	for data_point in range(len(C_norm)):
		F.append([])
		for i in range(6):
			for j in range(6):
				for k in range(6):
					F[data_point].append(B(i, C_norm[data_point][0])*B(j, C_norm[data_point][1])*B(k, C_norm[data_point][2]))

	# return numpy.linalg.lstsq(numpy.array(F), numpy.array(C_expected))[0], C_min, C_max
	return numpy.dot(numpy.linalg.pinv(numpy.array(F), rcond=.001), numpy.array(C_expected)), C_min, C_max

def correct_distortion(coeffs, q, q_min, q_max):
	G_corrected = []
	q_norm = []
	for i in range(len(q)):
		to_append = []
		# For each vector, take each component and normalize them
		for j in range(3):
			to_append.append((float(q[i][j]) - float(q_min[j]))/(float(q_max[j]) - float(q_min[j])))
		q_norm.append(to_append)


	F = []
	for data_point in range(len(q_norm)):
		F.append([])
		for i in range(6):
			for j in range(6):
				for k in range(6):
					F[data_point].append(B(i, q_norm[data_point][0])*B(j, q_norm[data_point][1])*B(k, q_norm[data_point][2]))
	F = numpy.array(F)
	G_corrected = numpy.dot(F, coeffs)
	# print(G_corrected.shape)
	return G_corrected

def get_frame_output(G, g):
    G = numpy.array(G)
    G_original = G
    g = numpy.array(g)
    g_original = g
    Gx, Gy, Gz = numpy.sum(G, axis=1)
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
    G = G - centroid_1 #center around origin

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



if (len(sys.argv) != 7):
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
F_d = []
F_a = []
C_expected = []
C = []

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
		line = cal_readings.readline().split(",") # this is unnecessary right now
		C.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		inside = numpy.dot(R_a, c[j]) + P_a
		square = numpy.dot(R_d_i, inside) - P_d_i
		C_expected.append([square[0][0], square[1][1], square[2][2]])

temp = numpy.array(C_expected)
C = numpy.array(C)

# print (numpy.allclose(temp, C, rtol=.01))

coeffs, q_min, q_max = get_coeff(temp, C)

q_corrected = correct_distortion(coeffs, C, q_min, q_max)

# print (numpy.allclose(q_corrected, temp, rtol=.01))


in_file = open(sys.argv[3])


frames = []
rotations = []
translations = []
G0 = []
g = []

first_line = in_file.readline().split(",")
num_markers = int(first_line[0].strip())
num_frames = int(first_line[1].strip())
for i in range(0, num_frames):
	G = []
	for j in range(0, num_markers):
		line = in_file.readline().split(",")
		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
		G.append(t)
	# print(numpy.array(G).T)
	# print(G)
	G = correct_distortion(coeffs, G, q_min, q_max)
	# print(G)
	G = G.T
	if i is 0:
		Gx, Gy, Gz = numpy.sum(G, axis=1)
		G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
		g = G - G0

	frames.append(get_frame(G, g))
	curr_rot = numpy.array(frames[i].get_rot())
	rotations.append([curr_rot[0][0], curr_rot[0][1], curr_rot[0][2], -1, 0, 0])
	rotations.append([curr_rot[1][0], curr_rot[1][1], curr_rot[1][2], 0, -1, 0])
	rotations.append([curr_rot[2][0], curr_rot[2][1], curr_rot[2][2], 0, 0, -1])
	t = -1*frames[i].get_trans()
	translations.append(t[0])
	translations.append(t[1])
	translations.append(t[2])

# for frame in frames:
# 	print(frame.get_rot())
a = numpy.squeeze(numpy.array(rotations))
b = numpy.array(translations)
x = numpy.linalg.lstsq(numpy.squeeze(numpy.array(rotations)), numpy.squeeze(numpy.array(translations)))
# print(numpy.array(x[0][3:6]))
# print(numpy.array(x[0]))




tG = numpy.array(x[0][0:3])
# print(tG)

# get EM fiducials
fiducials_file = open(sys.argv[4])

fid_first_line = fiducials_file.readline().split(",")
fid_markers = int(fid_first_line[0].strip())
fid_points = int(fid_first_line[1].strip())


# G0 = []
# g = []
fiducials = []
for i in range(0, fid_points):
	G = []
	for j in range(0, fid_markers):
		line = fiducials_file.readline().split(",")
		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
		G.append(t)
	G = numpy.array(G)
	G = correct_distortion(coeffs, G, q_min, q_max)
	G = G.T
	# if i is 0:
	# 	Gx, Gy, Gz = numpy.sum(G, axis=1)
	# 	G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
	# 	g = G - G0
	Fg = get_frame(G, g)
	fiducials.append(numpy.dot(Fg.get_rot(), tG) + Fg.get_trans().T)

print (numpy.array(fiducials))
# Extracting info from the extraneous array
fids = []
for fid in fiducials:
	fids.append(fid[0])

################################################
###Everything is probably correct up to here###
################################################
# get CT fiducials
ct_fid_file = open(sys.argv[5])
ct_fid_first_line = ct_fid_file.readline().split(",")
ct_fid_points = int(ct_fid_first_line[0].strip())
b = []
for i in range (ct_fid_points):
	line = ct_fid_file.readline().split(",")
	t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
	b.append(t)

# fids[i] = Bi and b[i] = bi

fids = numpy.array(fids)
b = numpy.array(b)
# b = numpy.array(list(reversed(b)))
# print(reversed(b))
# fids = numpy.array([[0, 0, 1], [-1, 1, 5], [2, 4, 2]])
# b = numpy.array([[-1, -1, -1], [-2, -2, -2], [-3, -3, -3]])
print(fids)
print(b)
# F_reg = get_frame(fids.T, b.T)

# F s.t. F*fids = b
# print(b.shape, fids.shape)
F_reg = get_frame_output(b.T, fids.T)

F_reg_rot = numpy.array(F_reg.get_rot())
F_reg_trans = numpy.array(F_reg.get_trans())

# print(F_reg_rot)
# print(F_reg_trans)

# for i in range(len(fids)):
# 	print(numpy.dot(F_reg_rot, fids[i]) + F_reg_trans.T)
# 	print(b[i])

# print(numpy.allclose(numpy.dot(F_reg_rot, b) + F_reg_trans.T, fids))



# get EM Nav
nav_file = open(sys.argv[6])
nav_first_line = nav_file.readline().split(",")
nav_markers = int(nav_first_line[0].strip())
nav_frames = int(nav_first_line[1].strip())

navs = []
for i in range(nav_frames):
	G = []
	for j in range(0, nav_markers):
		line = nav_file.readline().split(",")
		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
		G.append(t)
	G = numpy.array(G)
	G = correct_distortion(coeffs, G, q_min, q_max)
	G = G.T
	# if i is 0:
	# 	Gx, Gy, Gz = numpy.sum(G, axis=1)
	# 	G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
	# 	g = G - G0
	Fg = get_frame(G, g)
	navs.append(numpy.dot(Fg.get_rot(), tG) + Fg.get_trans().T)

navs = numpy.array(navs)
print(navs)
navs_extract = []
for nav in navs:
	navs_extract.append(nav[0])
for i in range(len(navs_extract)):
	print(numpy.dot(F_reg_rot, navs_extract[i]) + F_reg_trans.T)