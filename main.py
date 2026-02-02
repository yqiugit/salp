



#%%
#Simulation Setup

import numpy as np
import matplotlib.pyplot as plt
from model import JetPropulsion
from integration import rk_four

SIM_TIME = 15.0
T = 0.04

t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)


# %%
# Setup Model

bodyRadius = 6
radiusNozzle = 4.5

waterDens = 1.0
areaEject = np.pi * (radiusNozzle**2)
areaFront = np.pi * (bodyRadius**2) 
dragCoeff = 0.0

jetter = JetPropulsion(waterDens, areaEject, areaFront, dragCoeff)

# %%
#Simulate Open Loop System

x_init = np.zeros(2)
x_init[0] = 50.0
x_init[1] = 0.0 

x = np.zeros((2, N))
u = np.zeros((1, N))

x[:, 0] = x_init

for k in range(1,N):

    volume = (np.sin(k*T)**2)
    flow_rate = 2 * (np.sin(k*T) * np.cos(k*T))

    u[:,k] = flow_rate
    x[:,k] = rk_four(jetter.f, x[:, k-1], u[:, k-1], T)


# %%
# Plotting

plt.figure()
plt.plot(t, x[0, :], label="x₁")
plt.xlabel("Time [s]")
plt.ylabel("State x1: Mass")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, x[1, :], label="x₂")
plt.xlabel("Time [s]")
plt.ylabel("State x2: Velocity")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, u[0, :], label="u")
plt.xlabel("Time [s]")
plt.ylabel("Input")
plt.legend()
plt.grid(True)
plt.show()

# plt.figure()
# plt.plot(t, x[0, :], label="x₁: mass")
# plt.plot(t, x[1, :], label="x₂: velocity")
# plt.plot(t, u[0, :], "--", label="u: jet velocity")
# plt.xlabel("Time [s]")
# plt.legend(loc="upper right")
# plt.grid(True)
# plt.show()

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

# u: jet velocity
axs[0].plot(t, u[0, :], color="tab:green")
axs[0].set_ylabel("u: Volumetric Flow Rate")
axs[0].grid(True)

# x1: mass
axs[1].plot(t, x[0, :], color="tab:blue")
axs[1].set_ylabel("x₁: mass")
axs[1].grid(True)

# x2: velocity
axs[2].plot(t, x[1, :], color="tab:orange")
axs[2].set_ylabel("x₂: velocity")
axs[2].set_xlabel("Time [s]")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# %%
