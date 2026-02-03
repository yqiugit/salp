import numpy as np 


class JetPropulsion:
    
    def __init__(self, dens, areaEject, areaFront, dragCoeff):
        self.dens = dens
        self.areaEject = areaEject
        self.areaFront = areaFront
        self.dragCoeff = dragCoeff

    def f(self, x, u):

        f = np.zeros(3)

        f[0] = self.dens * u[0] 

        if u[0]<0:
            jet_vel = u[0]/self.areaEject
            thrust = (f[0]*jet_vel)
        else:
            thrust = -(f[0]*x[1])

        drag = -0.5 * self.dragCoeff * self.areaFront * (abs(x[1]) * x[1])

        f[1] = (1/x[0]) * (drag + thrust)

        f[2] = x[1]     #position

        return f


    



