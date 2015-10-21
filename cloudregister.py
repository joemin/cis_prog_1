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
		# print(numpy.array([tpose]).T)
		# D.append(numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())]).T)
		D.append(numpy.array(tpose).T)
	for j in range(N_a):
		line = cal_readings.readline().split(",")
		tpose = numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		A.append(numpy.array(tpose).T)
	# print(numpy.array(D).T)
	# print(numpy.array(D).T)
	F_d.append(get_frame(numpy.array(D).T, numpy.array(d).T))
	F_a.append(get_frame(numpy.array(A).T, numpy.array(a).T))
	# rod_d = superimposition_matrix(D, d, False, False)
	R_d_i = numpy.linalg.inv(F_d[i].get_rot())
	P_d_i = numpy.dot(R_d_i, F_d[i].get_trans())
	P_a = F_a[i].get_trans()
	R_a = F_a[i].get_rot()
	# print(new_trans)
	# for each frame, calculate Fd and Fa, and compute C expected using Fd-1 * Fa * c
	for j in range(N_c):
		line = cal_readings.readline().split(",") # this is unnecessary right now
		C.append([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())])
		inside = numpy.dot(R_a, c[j]) + P_a
		square = numpy.dot(R_d_i, inside) - P_d_i
		C_expected.append([square[0][0], square[1][1], square[2][2]])

print (numpy.allclose(numpy.array(C_expected), numpy.array(C), rtol=.01))
