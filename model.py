import numpy as np 


class Salp:
    
    def __init__(self, 
                 dens, 
                 areaEject, 
                 areaFront, 
                 areaSide, 
                 dragCoeff, 
                 nozzleLoss,
                 length, 
                 width, 
                 nozzleLength, 
                 intakeLength):

        self.rho = dens
        self.Ae = areaEject
        self.Af = areaFront
        self.As = areaSide
        self.Cd = dragCoeff
        self.Cn = nozzleLoss
        self.L = length
        self.W = width
        self.l1 = nozzleLength
        self.l2 = intakeLength


    def f(self, x, u):

        #unpack state anc action
        m, x_pos, y_pos, theta, vx, vy, omega = x
        dv, beta = u

        f = np.zeros(7)

        #mass flow rate (dm/dt)
        f[0] = self.rho * u[0] 

        #global kinematics
        f[1] = vx * np.cos(theta) - vy * np.sin(theta)
        f[2] = vx * np.sin(theta) + vy * np.cos(theta)
        f[3] = omega
        
        #dynamics and moments
        thrust_mag = ((self.Cn * self.rho) / self.Ae) * (dv**2) if dv < 0 else 0
        #thrust
        ftx = thrust_mag * np.cos(beta)
        fty = thrust_mag * np.sin(beta)

        #intake momentum loss
        finx = -self.rho * dv * vx if dv > 0 else 0
        finy = -self.rho * dv * vy if dv > 0 else 0

        #drag
        fdx = -0.5 * self.Cd * self.rho * self.Af * np.abs(vx) * vx
        fdy = -0.5 * self.Cd * self.rho * self.As * np.abs(vy) * vy

        #inertia
        I  = (1/5) * m * ((self.L/2)**2 + (self.W/2)**2)
        Idot = (1/5) * f[0] * ((self.L/2)**2 + (self.W/2)**2)

        #torques
        tau_t  = -self.l1 * fty 
        tau_in = self.l2 * finy 
        tau_d  = -(1/120) * self.rho * self.Cd * self.W * (self.L**4) * np.abs(omega) * omega 

        #accelerations
        f[4] = (1/m) * (ftx + fdx + finx) + omega * vy
        f[5] = (1/m) * (fty + fdy + finy) - omega * vx
        f[6] = (1/I) * ((tau_t+ tau_d + tau_in) - (Idot * omega))

        return f


    



