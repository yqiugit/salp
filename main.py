



#%%
#Simulation Setup

import numpy as np
import matplotlib.pyplot as plt
from model import JetPropulsion
from integration import rk_four

SIM_TIME = 10.0
T = 0.04

t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)


# %%
# Setup Model

waterDens = 1.0
areaEject = 9.0
areaFront = 10.0
dragCoeff = 0.05

jetter = JetPropulsion(waterDens, areaEject, areaFront, dragCoeff)

# %%
#Simulate Open Loop System

x_init = np.zeros(2)
x_init[0] = 5.0
x_init[1] = 0.0 

x = np.zeros((2, N))
u = np.zeros((1, N))

x[:, 0] = x_init

for k in range(1,N):

    x[:,k] = rk_four(jetter.f, x[:, k-1], u[:, k-1], T)

    u[:,k] = 5 * ((np.sin(0.1 * k))**2)


# %%
# Plotting

plt.figure()
plt.plot(t, x[0, :], label="x₁")
plt.plot(t, x[1, :], label="x₂")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, u[0, :], label="u")
plt.xlabel("Time [s]")
plt.ylabel("Input")
plt.legend()
plt.grid(True)
plt.show()

# %%
