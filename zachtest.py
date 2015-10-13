import numpy
import math
import sys
from transformations import superimposition_matrix, random_rotation_matrix, quaternion_matrix

_EPS = numpy.finfo(float).eps * 4.0

def get_frame(G, g):
    G = numpy.array(G)
    G_original = G
    g = numpy.array(g)
    g_original = g
    Gx, Gy, Gz = numpy.sum(G, axis=1)
    print(Gx, Gy, Gz)
    # print(Gx, Gy, Gz)
    # print(len(G))
    print(len(G[0]))
    centroid_1 = numpy.array([[Gx], [Gy], [Gz]])/len(G[0])
    G = G - centroid_1 #center around origin
    # print(centroid_1)

    gx, gy, gz = numpy.sum(g, axis=1)
    print(gx, gy, gz)
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
    print(t)
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

d = numpy.array([[-10,     0.00,     0.00],[5,     -250,   0.00],[5,   250.00,     0.00], [0.00,0.00,0.00]])
d = d.T
#D = [[208.68,   211.68, -1288.03],[211.93,   206.51, -1038.11],[213.65,   461.58, -1282.93]]

t = numpy.random.rand(3,1)
print(t)

R = random_rotation_matrix()
D = numpy.dot(R[:3,:3], d) + t

print(D)

# print(d)
# print(R[:3,:3])
#print(D)

f = get_frame(D,d)
# print(f.get_trans())
# print(f.get_rot())

#print(numpy.array(d))
print("***********************************")
print(numpy.dot(f.get_rot(), d) + f.get_trans())
print(D)