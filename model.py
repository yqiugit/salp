import numpy as np 


class JetPropulsion:
    
    def __init__(self, dens, areaEject, areaFront, dragCoeff):
        self.dens = dens
        self.areaEject = areaEject
        self.areaFront = areaFront
        self.dragCoeff = dragCoeff

    def f(self, x, u):

        f = np.zeros(2)
        f[0] = -u[0] * self.dens * self.areaEject
        f[1] = (1/x[0]) * ((-0.5 * self.dragCoeff * self.areaFront * (x[1]**2)) 
                           + ((u[0]**2) * self.dens * self.areaEject))
        return f
