



#%%
#Simulation Setup

import numpy as np
import matplotlib.pyplot as plt
from model import Salp
from integration import rk_four

SIM_TIME = 10.0
T = 0.002

t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)


# %%
# Setup Model

bodyWidth = 0.10
bodyLength = 0.4
radiusNozzle = 0.045
dragCoeff = 1.1
nozzleLoss = 0.95

waterDens = 1000.0   #kg/m^3
areaEject = np.pi * (radiusNozzle**2)
areaFront = np.pi * ((bodyWidth/2)**2) 
areaSide = np.pi * ((bodyLength/2) * (bodyWidth/2))
length2nozzle = bodyLength/2        # equal assuming COM at center
length2intake = bodyLength/2


salp = Salp(dens=waterDens, 
            areaEject=areaEject, 
            areaFront=areaFront, 
            areaSide=areaSide,
            dragCoeff=dragCoeff,
            nozzleLoss=nozzleLoss,
            length=bodyLength,
            width=bodyWidth,
            nozzleLength=length2nozzle,
            intakeLength=length2intake)

# %%
#Simulate Open Loop System

x_init = np.zeros(7)
x_init[0] = 0.5

x = np.zeros((7, N))
u = np.zeros((2, N))

x[:, 0] = x_init
freq_cmd = 0.5

for k in range(1,N):
    t_curr = (k-1) * T
    omega = 2*np.pi * freq_cmd

    flow_rate = 0.002 * omega * np.sin(2 * omega * t_curr)
    thrustAngle = 5

    u[0,k-1] = flow_rate
    u[1,k-1] = np.deg2rad(thrustAngle)
    x[:,k] = rk_four(salp.f, x[:, k-1], u[:, k-1], T)


# %%
# Plotting

fig, axs = plt.subplots(4, 2, figsize=(14, 12))
fig.tight_layout(pad=4.0)

# Input u1 (Volume Flow Rate)
# Now directly plots the u array you generated in the loop
axs[0, 0].plot(t, u[0, :], 'k--')
axs[0, 0].set_title('Input u1 (Volume Flow Rate)')
axs[0, 0].set_ylabel('m^3/s')

# x1: Mass
axs[0, 1].plot(t, x[0, :], 'b')
axs[0, 1].set_title('x1: Total Mass (m)')
axs[0, 1].set_ylabel('kg')

# x5: Vx
axs[1, 0].plot(t, x[4, :], 'r')
axs[1, 0].set_title('x5: Longitudinal Velocity (Vx)')
axs[1, 0].set_ylabel('m/s')

# x6: Vy
axs[1, 1].plot(t, x[5, :], 'g')
axs[1, 1].set_title('x6: Lateral Velocity (Vy)')
axs[1, 1].set_ylabel('m/s')

# x7: Omega
axs[2, 0].plot(t, x[6, :], 'm')
axs[2, 0].set_title('x7: Angular Velocity (Omega)')
axs[2, 0].set_ylabel('rad/s')

# x4: Heading
axs[2, 1].plot(t, np.rad2deg(x[3, :]), 'c')
axs[2, 1].set_title('x4: Heading Angle (Theta)')
axs[2, 1].set_ylabel('Degrees')

# Global XY Trajectory
axs[3, 0].plot(x[1, :], x[2, :], 'k')
axs[3, 0].set_title('Global XY Trajectory')
axs[3, 0].set_xlabel('X (m)')
axs[3, 0].set_ylabel('Y (m)')
axs[3, 0].axis('equal')  # Ensures the spatial plot isn't stretched

# Hide unused subplot
axs[3, 1].axis('off')

plt.show()


# %%
#Animate

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from IPython.display import HTML

def animate_salp(x_data, t_data, interval=20):
    # Setup the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set plot limits with a buffer around the trajectory
    padding = 0.1
    ax.set_xlim(np.min(x_data[1, :]) - padding, np.max(x_data[1, :]) + padding)
    ax.set_ylim(np.min(x_data[2, :]) - padding, np.max(x_data[2, :]) + padding)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Salp Robot Trajectory Animation")

    # Trace line for the past trajectory
    trace, = ax.plot([], [], 'k-', alpha=0.3, lw=1)
    
    # The Robot Hull (Ellipse)
    # x[3] is theta in radians
    hull = Ellipse((0, 0), width=bodyLength, height=bodyWidth, 
                   edgecolor='blue', facecolor='lightblue', lw=2)
    ax.add_patch(hull)

    # Orientation Arrow (Nose of the robot)
    arrow_len = 0.05
    nose, = ax.plot([], [], 'r-', lw=2)

    def init():
        trace.set_data([], [])
        nose.set_data([], [])
        return hull, trace, nose

    def update(frame):
        # Current States
        curr_x = x_data[1, frame]
        curr_y = x_data[2, frame]
        curr_theta = np.rad2deg(x_data[3, frame])
        
        # Update Hull Position and Rotation
        hull.set_center((curr_x, curr_y))
        hull.set_angle(curr_theta)
        
        # Update Trajectory Trace
        trace.set_data(x_data[1, :frame], x_data[2, :frame])
        
        # Update Orientation Arrow
        theta_rad = x_data[3, frame]
        nx = [curr_x, curr_x + arrow_len * np.cos(theta_rad)]
        ny = [curr_y, curr_y + arrow_len * np.sin(theta_rad)]
        nose.set_data(nx, ny)
        
        return hull, trace, nose

    # Create Animation
    # Using 'step' to downsample if N is large (e.g., plot every 10th frame)
    step = 10 
    frames = range(0, len(t_data), step)
    
    ani = FuncAnimation(fig, update, frames=frames, 
                        init_func=init, blit=True, interval=interval)
    
    plt.show()
    return ani

# Call the function with your simulation results
ani = animate_salp(x, t)

ani.save("salp_trajectory.gif", writer='pillow', fps=30)

HTML(ani.to_jshtml())

# %%
