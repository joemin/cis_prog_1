import numpy
import math
import sys
from transformations import superimposition_matrix, random_rotation_matrix, quaternion_matrix

_EPS = numpy.finfo(float).eps * 4.0

def get_frame(G, g):
    G = numpy.array(G)
    G_original = G
    g = numpy.array(g)
    G_original = g
    Gx, Gy, Gz = numpy.sum(G, axis=1)
    # print(Gx, Gy, Gz)
    # print(len(G))
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G)
    G = G - centroid_1 #center around origin
    # print()
    # print(G)
    # print()
    # print(centroid_1)

    gx, gy, gz = numpy.sum(g, axis=1)
    centroid_2 = numpy.array([[gx], [gy], [gz]])/len(g)
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

    t = numpy.dot(R, centroid_2) - centroid_1
    if (t[0][0] < .00000000001):
        t[0][0] = 0.00
    if (t[1][0] < .00000000001):
        t[1][0] = 0.00
    if (t[2][0] < .00000000001):
        t[2][0] = 0.00

    return Frame(R.T, t)


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


"""if (len(sys.argv) != 3):
    sys.exit(0)"""
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

d = numpy.array([[-10,     0.00,     0.00],[5,     -250,   0.00],[5,   250.00,     0.00], [0.00,0.00,0.00]])
d = d.T
#D = [[208.68,   211.68, -1288.03],[211.93,   206.51, -1038.11],[213.65,   461.58, -1282.93]]

R = random_rotation_matrix()
D = numpy.dot(R[:3,:3], d)

# print(D)

# print(d)
print(R[:3,:3])
#print(D)

f = get_frame(D,d)
print(f.get_trans())
print(f.get_rot())

#print(numpy.array(d))
print("***********************************")
print(numpy.dot(f.get_rot(), d) + f.get_trans())
print(D)