import numpy as np 


class JetPropulsion:
    
    def __init__(self, dens, areaEject, areaFront, dragCoeff):
        self.dens = dens
        self.areaEject = areaEject
        self.areaFront = areaFront
        self.dragCoeff = dragCoeff

    def f(self, x, u):

        f = np.zeros(2)

        f[0] = self.dens * u[0] 

        if u[0]<0:
            jet_vel = u[0]/self.areaEject
            thrust = (f[0]*jet_vel)
        else:
            thrust = -(f[0]*x[1])

        f[1] = (1/x[0]) * ((-0.5 * self.dragCoeff * self.areaFront * (x[1]**2)) 
                           + thrust)
        return f

    def volume(self, t, crankRadius, angularW, crankLength, nozzleRadius, height):  

        sqrt_term = np.sqrt(np.maximum(crankLength**2 - (crankRadius * np.sin(angularW * t))**2, 0))
        s = crankRadius * np.cos(angularW * t) + sqrt_term - crankLength

        num = -nozzleRadius**2 - 4 * nozzleRadius * s + 4 * s**2
        den = 2 * nozzleRadius**2
        alpha = np.arccos(np.clip(num / den, -1, 1))

        C = (nozzleRadius**2 / 2) * (alpha - np.sin(alpha))
        A = np.pi * nozzleRadius**2 - 2 * C
        
        return A * height
    
    



