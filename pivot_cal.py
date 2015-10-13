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
    # print(len(G))
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
    G = G - centroid_1 #center around origin
    # print(centroid_1)

    gx, gy, gz = numpy.sum(g, axis=1)
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

if (len(sys.argv) != 2):
	sys.exit(0)

in_file = open(sys.argv[1])

# frames = numpy.array([])
# rotations = numpy.array([])
# translations = numpy.array([])
frames = []
rotations = []
translations = []
G0 = []

first_line = in_file.readline().split(",")
num_markers = int(first_line[0].strip())
num_frames = int(first_line[1].strip())
print(num_frames)
for i in range(0, num_frames):
# for i in range(0, 1):
	G = []
	for j in range(0, num_markers):
		line = in_file.readline().split(",")
		# get array G1
		t = [float(line[0].strip()),float(line[1].strip()), float(line[2].strip())]
		G.append(t)
		# calculate G0
	G = numpy.array(G).T
	if i is 0:
		Gx, Gy, Gz = numpy.sum(G, axis=1)
		G0 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
	# calculate g
	g = G - G0
	print(G)
	frames.append(get_frame(G, g))
	# print(numpy.dot(frames[-1].get_rot(), g) + frames[-1].get_trans())
	# print(G)
	curr_rot = numpy.array(frames[i].get_rot())
	# rotations.append([[curr_rot[0][0], curr_rot[0][1], curr_rot[0][2], -1, 0, 0], [curr_rot[1][0], curr_rot[1][1], curr_rot[1][2], 0, -1, 0], [curr_rot[2][0], curr_rot[2][1], curr_rot[2][2], 0, 0, -1]])
	rotations.append([curr_rot[0][0], curr_rot[0][1], curr_rot[0][2], -1, 0, 0])
	rotations.append([curr_rot[1][0], curr_rot[1][1], curr_rot[1][2], 0, -1, 0])
	rotations.append([curr_rot[2][0], curr_rot[2][1], curr_rot[2][2], 0, 0, -1])
	t = -1*frames[i].get_trans()
	translations.append(t[0])
	translations.append(t[1])
	translations.append(t[2])
	#print(translations)


# # solve Pdimple = frames[k]*t
# print(numpy.array(rotations))
# print(numpy.array(translations))
a = numpy.squeeze(numpy.array(rotations))
b = numpy.array(translations)
# print(a)
# print(b)
x = numpy.linalg.lstsq(numpy.squeeze(numpy.array(rotations)), numpy.squeeze(numpy.array(translations)))
# print(numpy.array(x))
# print(a * x)
# print(b)