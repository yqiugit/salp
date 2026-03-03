



#%%
#Simulation Setup

import numpy as np
import matplotlib.pyplot as plt
from model import JetPropulsion
from integration import rk_four

SIM_TIME = 40.0
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

x_init = np.zeros(3)
x_init[0] = 50.0
x_init[1] = 0.0 
x_init[2] = 0.0

x = np.zeros((3, N))
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
plt.plot(t, x[2, :], label="x₂")
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

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

# u: jet velocity
axs[0].plot(t, u[0, :], color="tab:green")
axs[0].set_ylabel("u: Volumetric Flow Rate")
axs[0].grid(True)

# x1: mass
axs[1].plot(t, x[0, :], color="tab:blue")
axs[1].set_ylabel("x₁: mass")
axs[1].grid(True)

axs[2].plot(t, x[2, :], color="tab:purple")
axs[2].set_ylabel("x1: Position")
axs[2].grid(True)

# x2: velocity
axs[3].plot(t, x[1, :], color="tab:orange")
axs[3].set_ylabel("x2: velocity")
axs[3].set_xlabel("Time [s]")
axs[3].grid(True)



plt.tight_layout()
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def salp_dynamics(t, x, params):
    # Unpack states
    m, x_pos, y_pos, theta, vx, vy, omega = x
    
    # Extract parameters
    rho, Ae, Af, As, Cd, L, W, l1, l2 = params['rho'], params['Ae'], params['Af'], params['As'], params['Cd'], params['L'], params['W'], params['l1'], params['l2']
    
    # Define Inputs
    # u1: Sinusoidal pumping (amplitude = 0.002 m^3/s, freq = 1 Hz)
    u1 = 0.002 * np.sin(2 * np.pi * 1.0 * t) 
    # u2: Constant 20-degree nozzle angle
    u2 = np.deg2rad(0) 
    
    # 1. Mass Dynamics
    x1_dot = rho * u1
    
    # 2. Global Kinematics
    x2_dot = vx * np.cos(theta) - vy * np.sin(theta)
    x3_dot = vx * np.sin(theta) + vy * np.cos(theta)
    x4_dot = omega
    
    # 3. Forces & Moments
    thrust_mag = (rho / Ae) * (u1**2) if u1 < 0 else 0
    ftx = thrust_mag * np.cos(u2)
    fty = thrust_mag * np.sin(u2)
    
    finx = -rho * u1 * vx if u1 > 0 else 0
    finy = -rho * u1 * vy if u1 > 0 else 0
    
    fdx = -0.5 * Cd * rho * Af * np.abs(vx) * vx
    fdy = -0.5 * Cd * rho * As * np.abs(vy) * vy
    
    # Inertia
    I  = (1/5) * m * ((L/2)**2 + (W/2)**2)
    Idot = (1/5) * x1_dot * ((L/2)**2 + (W/2)**2)
    
    # Torques
    tau_t  = -l1 * fty 
    tau_in = l2 * finy 
    tau_d  = -(1/120) * rho * Cd * W * (L**4) * np.abs(omega) * omega 
    
    # 4. Accelerations
    x5_dot = (1/m) * (fdx + ftx + finx - x1_dot * vx) + omega * vy
    x6_dot = (1/m) * (fdy + fty + finy - x1_dot * vy) - omega * vx
    x7_dot = (1/I) * ((tau_t + tau_d + tau_in) - Idot * omega)
    
    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot]

# --- Simulation Setup ---
# Physical Parameters (Adjust these to match your Onshape model)
params = {
    'rho': 1000.0, 'Ae': 0.002, 'Af': 0.03, 'As': 0.08, 'Cd': 1.1,
    'L': 0.4, 'W': 0.15, 'l1': 0.2, 'l2': 0.2
}

# Initial Conditions: [m, X, Y, theta, Vx, Vy, omega]
# Note: Initial mass includes hull + starting water
x0 = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Time vector (Simulate for 5 seconds to see 5 pump cycles)
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Run ODE Solver
sol = solve_ivp(salp_dynamics, t_span, x0, t_eval=t_eval, args=(params,), method='RK45')

# --- Plotting ---
fig, axs = plt.subplots(4, 2, figsize=(14, 12))
fig.tight_layout(pad=4.0)

# Input u1
axs[0, 0].plot(sol.t, 0.002 * np.sin(2 * np.pi * 1.0 * sol.t), 'k--')
axs[0, 0].set_title('Input u1 (Volume Flow Rate)')
axs[0, 0].set_ylabel('m^3/s')

# x1: Mass
axs[0, 1].plot(sol.t, sol.y[0], 'b')
axs[0, 1].set_title('x1: Total Mass (m)')
axs[0, 1].set_ylabel('kg')

# x5: Vx
axs[1, 0].plot(sol.t, sol.y[4], 'r')
axs[1, 0].set_title('x5: Longitudinal Velocity (Vx)')
axs[1, 0].set_ylabel('m/s')

# x6: Vy
axs[1, 1].plot(sol.t, sol.y[5], 'g')
axs[1, 1].set_title('x6: Lateral Velocity (Vy)')
axs[1, 1].set_ylabel('m/s')

# x7: Omega
axs[2, 0].plot(sol.t, sol.y[6], 'm')
axs[2, 0].set_title('x7: Angular Velocity (Omega)')
axs[2, 0].set_ylabel('rad/s')

# x4: Heading
axs[2, 1].plot(sol.t, np.rad2deg(sol.y[3]), 'c')
axs[2, 1].set_title('x4: Heading Angle (Theta)')
axs[2, 1].set_ylabel('Degrees')

# Global XY Trajectory
axs[3, 0].plot(sol.y[1], sol.y[2], 'k')
axs[3, 0].set_title('Global XY Trajectory')
axs[3, 0].set_xlabel('X (m)')
axs[3, 0].set_ylabel('Y (m)')
axs[3, 0].axis('equal')

# Hide unused subplot
axs[3, 1].axis('off')

plt.show()



# %%
