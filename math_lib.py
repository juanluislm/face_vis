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
