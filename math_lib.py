import numpy as np
from pyquaternion import Quaternion
import cv2
import math

def translate(position):
    return np.array([[1,0,0,position[0]], [0,1,0,position[1]], [0,0,1,position[2]], [0,0,0,1]])

def scale(scale):
    return np.array([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]])

def rotate(rotate_in_degrees):
    qX = Quaternion(axis=[1, 0, 0], angle=rotate_in_degrees[0]/180*math.pi)
    qY = Quaternion(axis=[0, 1, 0], angle=rotate_in_degrees[1]/180*math.pi)
    qZ = Quaternion(axis=[0, 0, 1], angle=rotate_in_degrees[2]/180*math.pi)
    qTotal = qZ * qY * qX
    return qTotal.transformation_matrix

def rotate2(rotate_in_degrees):
    rotateX, _ = cv2.Rodrigues(np.array([rotate_in_degrees[0]/180*math.pi, 0, 0]))
    rotateY, _ = cv2.Rodrigues(np.array([0, rotate_in_degrees[1]/180*math.pi, 0]))
    rotateZ, _ = cv2.Rodrigues(np.array([0, 0, rotate_in_degrees[2]/180*math.pi]))
    return np.dot(np.dot(rotateZ, rotateY), rotateX)

def isRotationMatrix(m):
    mt = np.transpose(m)
    shouldBeIdentity = np.dot(mt, m)
    I = np.identity(4, dtype = m.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAnglesInDegrees(m) :
    assert(isRotationMatrix(m))
    sy = math.sqrt(m[0,0] * m[0,0] +  m[1,0] * m[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(m[2,1] , m[2,2])
        y = math.atan2(-m[2,0], sy)
        z = math.atan2(m[1,0], m[0,0])
    else :
        x = math.atan2(-m[1,2], m[1,1])
        y = math.atan2(-m[2,0], sy)
        z = 0
    return np.array([x/math.pi*180, y/math.pi*180, z/math.pi*180])

def transform_to_translate_rotation_scale(m):
    a, b, c, d = m[0]
    e, f, g, h = m[1]
    i, j, k, l = m[2]
    translateM = translate([d, h, l])
    sx = np.linalg.norm([a, e, i])
    sy = np.linalg.norm([b, f, j])
    sz = np.linalg.norm([c, g, k])
    rotateM = np.array([[a/sx, b/sy, c/sz, 0],
                        [e/sx, f/sy, g/sz, 0],
                        [i/sx, j/sy, k/sz, 0],
                        [0,    0,    0,    1]])
    scaleM = scale([sx, sy, sz])
    return translateM, rotateM, scaleM

def translateM_to_vector(m):
    return np.array([m[0][3], m[1][3], m[2][3]])

def rotateM_to_vector(m):
    return rotationMatrixToEulerAnglesInDegrees(m)

def scalingM_to_vector(m):
    return np.array([m[0][0]-1, m[1][1]-1, m[2][2]-1])
