import numpy
import math
import sys

def get_frame(G, g):
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
	centroid_1 = numpy.dot(R, numpy.array([Gx, Gy, Gz])/len(G))

	gx, gy, gz = numpy.sum(g, axis=0)
	centroid_2 = numpy.array([gx, gy, gz])/len(g)

	t = centroid_1 - centroid_2
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

if (len(sys.argv) != 2):
	sys.exit(0)

in_file = open(sys.argv[1])

# frames = numpy.array([])
# rotations = numpy.array([])
# translations = numpy.array([])
frames = []
rotations = []
translations = []

first_line = in_file.readline().split(",")
num_markers = int(first_line[0].strip())
num_frames = int(first_line[1].strip())

for i in range(0, num_frames):
	line = in_file.readline().split(",")
	G = []
	for j in range(0, num_markers):
		# get array G1
		G.append(numpy.array([float(line[0].strip()), float(line[1].strip()), float(line[2].strip())]))
		# calculate G0
	Gx, Gy, Gz = numpy.sum(G, axis=0)
	if i is 0:
		G0 = numpy.array([Gx, Gy, Gz])/len(G)
	# calculate g
	g = G - G0
	frames.append(get_frame(G, g))
	rotations.append(numpy.array([frames[i].get_rot(), -1*numpy.identity(3)]))
	translations.append(-1*numpy.array(frames[i].get_trans()).T)
	print(translations[i])
	# print(translations)


# numpy.append(frames, get_frame(G, g))
# print(frames)
# numpy.append(rotations, numpy.array([frames[0].get_rot(), -1*numpy.identity(3)]))
# numpy.append(translations, frames[0].get_trans())

# while(file is not empty):
# 	k++
# 	G = numpy.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# 	g = G - G0
# 	frames.append(get_frame(G, g))
# 	rotations.append([frames[k].get_rot(), -1*numpy.identity(3)])
# 	translations.append(frames[k].get_trans())


# solve Pdimple = frames[k]*t
# print(numpy.array(rotations))
# print(numpy.array(translations))
x = numpy.linalg.solve(numpy.array(rotations), numpy.array(translations))
# print(x)
# print(y)

