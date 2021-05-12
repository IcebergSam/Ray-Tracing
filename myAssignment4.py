import operator
from math import *
import numpy as np
from matrix import matrix

class cameraMatrix:

    def __init__(self, window, UP, E, G, nearPlane=10.0, farPlane=50.0, theta=90.0):
        self.__UP = UP.normalize()
        self.__E = E
        self.__G = G
        self.__np = nearPlane
        self.__fp = farPlane
        self.__width = window.getWidth()
        self.__height = window.getHeight()
        self.__theta = theta
        self.__aspect = self.__width / self.__height
        self.__npHeight = self.__np * (pi / 180.0 * self.__theta / 2.0)
        self.__npWidth = self.__npHeight * self.__aspect

        Mp = self.__setMp(self.__np, farPlane)
        T1 = self.__setT1(self.__np, self.__theta, self.__aspect)
        S1 = self.__setS1(self.__np, self.__theta, self.__aspect)
        T2 = self.__setT2()
        S2 = self.__setS2(self.__width, self.__height)
        W2 = self.__setW2(self.__height)

        self.__N = (self.__E - self.__G).removeRow(3).normalize()
        self.__U = self.__UP.removeRow(3).crossProduct(self.__N).normalize()
        self.__V = self.__N.crossProduct(self.__U)

        self.__Mv = self.__setMv(self.__U, self.__V, self.__N, self.__E)
        self.__C = W2 * S2 * T2 * S1 * T1 * Mp
        self.__M = self.__C * self.__Mv

    def __setMv(self, U, V, N, E):
        Mv = matrix(np.identity(4))
        Mv.set(0, 0, U.get(0, 0))
        Mv.set(0, 1, U.get(1, 0))
        Mv.set(0, 2, U.get(2, 0))
        Mv.set(0, 3, -E.removeRow(3).dotProduct(U))

        Mv.set(1, 0, V.get(0, 0))
        Mv.set(1, 1, V.get(1, 0))
        Mv.set(1, 2, V.get(2, 0))
        Mv.set(1, 3, -E.removeRow(3).dotProduct(V))

        Mv.set(2, 0, N.get(0, 0))
        Mv.set(2, 1, N.get(1, 0))
        Mv.set(2, 2, N.get(2, 0))
        Mv.set(2, 3, -E.removeRow(3).dotProduct(N))
        return Mv

    def __setMp(self, nearPlane, farPlane):
        Mp = matrix(np.identity(4))
        Mp.set(0, 0, nearPlane)
        Mp.set(1, 1, nearPlane)
        Mp.set(2, 2, -(farPlane + nearPlane) / (farPlane - nearPlane))
        Mp.set(2, 3, -2.0 * (farPlane * nearPlane) / (farPlane - nearPlane))
        Mp.set(3, 2, -1.0)
        Mp.set(3, 3, 0.0)
        return Mp

    def __setT1(self, nearPlane, theta, aspect):
        top = nearPlane * tan(pi / 180.0 * theta / 2.0)
        right = aspect * top
        bottom = -top
        left = -right
        T1 = matrix(np.identity(4))
        T1.set(0, 3, -(right + left) / 2.0)
        T1.set(1, 3, -(top + bottom) / 2.0)
        return T1

    def __setS1(self, nearPlane, theta, aspect):
        top = nearPlane * tan(pi / 180.0 * theta / 2.0)
        right = aspect * top
        bottom = -top
        left = -right
        S1 = matrix(np.identity(4))
        S1.set(0, 0, 2.0 / (right - left))
        S1.set(1, 1, 2.0 / (top - bottom))
        return S1

    def __setT2(self):
        T2 = matrix(np.identity(4))
        T2.set(0, 3, 1.0)
        T2.set(1, 3, 1.0)
        return T2

    def __setS2(self, width, height):
        S2 = matrix(np.identity(4))
        S2.set(0, 0, width / 2.0)
        S2.set(1, 1, height / 2.0)
        return S2

    def __setW2(self, height):
        W2 = matrix(np.identity(4))
        W2.set(1, 1, -1.0)
        W2.set(1, 3, height)
        return W2

    def getRay(self, window, i, j):
        a = -self.__np
        b = self.__npWidth * (2.0 * i / window.getWidth() - 1.0)
        c = self.__npHeight * (2.0 * (window.getHeight() - (j + 1)) / window.getHeight() - 1.0)
        return (self.__N.scalarMultiply(a) + self.__U.scalarMultiply(b) + self.__V.scalarMultiply(c)).insertRow(3, 0.0)

    """
    module: method: minimumIntersection(direction_vector_ray, objectList)
    author: Sam Ahsan
    date of creation: Apr 3/2020
    purpose: sort intersection points such that the object closest to the camera is ray-traced and
        objects behind it are not.
    parameters: direction_vector_ray, list_of_objects_in_the_scene
    output: return the generated sorted list of intersections
    explanation: this method generates a list of tuples (k, t_min), where
        k is the index of an object intersected by the direction_vector ray 
            in the list of objects composing the scene, and
        t_min is the minimum t-value of all intersections made by the ray with the object k,
    such that the list of tuples is sorted in order of increasing t_min
    """
    def minimumIntersection(self, direction, objectList):
        intersectionList = []

        for object in objectList:
            M_inv = object.getTinv()

            # transform t_e and t_d into viewing coordinates
            T_e = M_inv * self.__E
            T_d = M_inv * direction

            # compute intersection of direction ray with object
            t_val = object.intersection(T_e, T_d)

            if t_val != -1.0:
                intersectionList.append((objectList.index(object), t_val))

            intersectionList = sorted(intersectionList, key=lambda t_min: t_min[1])

        return intersectionList

    def worldToViewingCoordinates(self, P):
        return self.__Mv * P

    def worldToImageCoordinates(self, P):
        return self.__M * P

    def worldToPixelCoordinates(self, P):
        return self.__M * P.scalarMultiply(1.0 / (self.__M * P).get(3, 0))

    def viewingToImageCoordinates(self, P):
        return self.__C * P

    def viewingToPixelCoordinates(self, P):
        return self.__C * P.scalarMultiply(1.0 / (self.__C * P).get(3, 0))

    def imageToPixelCoordinates(self, P):
        return P.scalarMultiply(1.0 / P.get(3, 0))

    def getUP(self):
        return self.__UP

    def getU(self):
        return self.__U

    def getV(self):
        return self.__V

    def getN(self):
        return self.__N

    def getE(self):
        return self.__E

    def getG(self):
        return self.__G

    def getMv(self):
        return self.__Mv

    def getC(self):
        return self.__C

    def getM(self):
        return self.__M

    def getNp(self):
        return self.__np

    def getFp(self):
        return self.__fp

    def getTheta(self):
        return self.__theta

    def getAspect(self):
        return self.__aspect

    def getWidth(self):
        return self.__width

    def getHeight(self):
        return self.__height

    def getNpHeight(self):
        return self.__npHeight

    def getNpWidth(self):
        return self.__npWidth


"""
module: class: shader 
author: Sam Ahsan
date of creation: Apr 3/2020
purpose: shade objects in the scene according to light source
explanation: this class computes the (R,G,B) value for a pixel (i,j) in a graphicsWindow according
    to the position of i) the light source in the scene and ii) the camera, where the (R,G,B) value 
    of the shaded pixel is determined by the colour and reflectance of the object and the intensity 
    of the light source. 
"""
class shader:

    """
    module: __shadowed(object, intersection, vector_to_light_source, objectList)
    author: Sam Ahsan
    date of creation: Apr 3/2020
    purpose: helper method determines if the given intersection point on an object
        lies in the shadow of another object
    parameters: object, intersection_point, vector_to_light_source, objectLsist
    output: return True if point is shadowed; return False otherwise
    explanation: the given intersection_point and vector_to_light_source are used to test if the ray f
        rom the intersection point to the light source intersects any other objects in the scene, given
        by the objectList
    """
    def __shadowed(self, object, I, S, objectList):
        M = object.getT()

        # detach intersection point I from its surface,
        # then transform I and S into world coordinates
        I = M * (I + S.scalarMultiply(0.001))
        S = M * S

        for obj in objectList:
            # compute intersection point in object coordinates
            M_inv = obj.getTinv()
            I = M_inv * I
            S = (M_inv * S).normalize()

            if obj.intersection(I, S) != -1.0:
                # if this line is reached, an intersection exists with another object
                return True

        # if this line is reached, no intersection exists with any other object
        return False

    """
    module: constructor: __init__(intersection, direction, camera, objectList, light)
    author: Sam Ahsan
    date of creation: Apr 3/2020
    purpose: initialise new shader object
    parameters: tuple_from_intersectionList, direction_vector, camera, objectList, light
    output: compute the colour of the shaded pixel (i,j) as instance variable self.__color
    explanation: given the tuple associating a t-value with an object, compute the intersection point I
       and the vector from the intersection point to the light source position S, then use the specular
       reflection vector, the object colour, and the intensity of the light to compute the pixel shade
    """
    def __init__(self, intersection, direction, camera, objectList, light):
        object = objectList[intersection[0]]
        t_val = intersection[1]

        M_inv = object.getTinv()

        # transform to object coordinates
        T_s = M_inv * light.getPosition()
        T_e = M_inv * camera.getE()
        T_d = M_inv * direction

        # compute intersection point
        I = T_e + (T_d.scalarMultiply(t_val))

        # compute vector from intersection point to light source
        S = (T_s - I).normalize()

        N = object.normalVector(I)

        # compute specular reflection vector
        R = S.scalarMultiply(-1) + N.scalarMultiply((S.scalarMultiply(2)).dotProduct(N))

        # compute vector to center of projection
        V = (T_e - I).normalize()

        # compute diffuse and specular components of light
        I_d = max(N.dotProduct(S), 0)
        I_s = max(R.dotProduct(V), 0)

        r = object.getReflectance()
        c = object.getColor()
        L_i = light.getIntensity()

        # compute shade
        if not self.__shadowed(object, I, S, objectList):
            f = r[0] + r[1] * I_d + r[2] * (I_s ** r[3])
        else:
            f = r[0]

        self.__color = (int(f * c[0] * L_i[0]), int(f * c[1] * L_i[1]), int(f * c[2] * L_i[2]))

    def getShade(self):
        return self.__color
