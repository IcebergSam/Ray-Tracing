class shader:

    def __shadowed(self,object,I,S,objectList):
         #Complete this helper method
        M = object.getT()
        I = M * (I + S.scalarMultiply(0.001))
        S = M * S

        for obj in objectList:
            M_inv = obj.getTinv()
            I = M_inv * I
            S = (M_inv * S).normalize()

            if obj.intersection(I,S) != -1.0:
                return True

        return False

    def __init__(self,intersection,direction,camera,objectList,light):

	    #Complete this method
        object = objectList[intersection[0]]
        t_val = intersection[1]

        M_inv = object.getTinv()

        T_s = M_inv * light.getPosition()
        T_e = M_inv * camera.getE()
        T_d = M_inv * direction

        I = T_e + (T_d.scalarMultiply(t_val))

        S = (T_s - I).normalize()
        N = object.normalVector(I)

        R = S.scalarMultiply(-1) + N.scalarMultiply((S.scalarMultiply(2)).dotProduct(N))
        V = (T_e - I).normalize()

        I_d = max(N.dotProduct(S), 0)
        I_s = max(R.dotProduct(V), 0)

        r = object.getReflectance()
        c = object.getColor()
        L_i = light.getIntensity()

        if not self.__shadowed(object,I,S,objectList):
            f = r[0] + r[1]*I_d + r[2]*(I_s**r[3])
        else:
            f = r[0]

        self.__color =  (int(f*c[0]*L_i[0]),int(f*c[1]*L_i[1]),int(f*c[2]*L_i[2]))

    def getShade(self):
        return self.__color