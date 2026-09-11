"""
Navier-Stokes Learning Project

A beginner exploration of fluid motion using Python.

This project demonstrates:
1. A 2D fluid grid
2. Fluid spreading
3. Fluid movement
4. Flow around an obstacle
5. Velocity fields
6. Pressure fields
7. Viscosity

Note:
This is an educational model inspired by concepts from fluid dynamics.
It is NOT a full numerical solution of the Navier-Stokes equations.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. BASIC 2D FLUID GRID
# ============================================================

N = 50

fluid = np.zeros((N, N))

plt.imshow(fluid, origin="lower")
plt.title("Empty 2D Fluid Grid")
plt.colorbar()
plt.show()

# Put a small amount of fluid in the center
fluid[25, 25] = 1

plt.imshow(fluid, origin="lower")
plt.title("A Small Amount of Fluid")
plt.colorbar()
plt.show()


# ============================================================
# 2. FLUID SPREADING
# ============================================================

new_fluid = fluid.copy()

for i in range(1, N - 1):
    for j in range(1, N - 1):
        new_fluid[i, j] = (
            fluid[i, j]
            + fluid[i + 1, j]
            + fluid[i - 1, j]
            + fluid[i, j + 1]
            + fluid[i, j - 1]
        ) / 5

fluid = new_fluid

plt.imshow(fluid, origin="lower")
plt.title("Fluid Spreading")
plt.colorbar()
plt.show()


# ============================================================
# 3. FLUID VELOCITY
# ============================================================

fluid = np.zeros((N, N))

# Put fluid in the center
fluid[25, 25] = 1

# Horizontal velocity
velocity_x = np.zeros((N, N))

# Vertical velocity
velocity_y = np.zeros((N, N))

# Give the fluid a horizontal velocity
velocity_x[:, :] = 1

print("Grid size:", fluid.shape)
print("Fluid at center:", fluid[25, 25])
print("Horizontal velocity at center:", velocity_x[25, 25])
print("Vertical velocity at center:", velocity_y[25, 25])


# ============================================================
# 4. FLUID MOVING TO THE RIGHT
# ============================================================

fluid = np.zeros((50, 50))
fluid[25, 10] = 1

for step in range(20):
    fluid = np.roll(fluid, 1, axis=1)

    # Remove fluid that leaves the left boundary
    fluid[:, 0] = 0

    plt.imshow(fluid, origin="lower")
    plt.title(f"Fluid Moving to the Right - Step {step + 1}")
    plt.colorbar()
    plt.show()


# ============================================================
# 5. CONTINUOUS 2D FLUID FLOW
# ============================================================

N = 80
fluid = np.zeros((N, N))

for step in range(80):
    # Add fluid at the left side
    fluid[:, 2] = 1.0

    # Move the fluid to the right
    fluid = np.roll(fluid, 1, axis=1)

    # Remove fluid at the boundary
    fluid[:, 0] = 0

    plt.imshow(fluid, origin="lower")
    plt.title(f"2D Fluid Flow - Step {step + 1}")
    plt.colorbar()
    plt.show()

    time.sleep(0.05)


# ============================================================
# 6. FLUID FLOW AROUND AN OBSTACLE
# ============================================================

N = 80
fluid = np.zeros((N, N))

# Create an obstacle
obstacle = np.zeros((N, N), dtype=bool)
obstacle[25:55, 40] = True

for step in range(80):
    # Add incoming fluid
    fluid[:, 2] = 1.0

    # Remove fluid inside the obstacle
    fluid[obstacle] = 0

    # Move the fluid
    fluid = np.roll(fluid, 1, axis=1)

    # Boundary condition
    fluid[:, 0] = 0

    plt.imshow(fluid, origin="lower")
    plt.title(f"Fluid Flow Around an Obstacle - Step {step + 1}")
    plt.colorbar()
    plt.show()

    time.sleep(0.05)


# ============================================================
# 7. VELOCITY FIELD
# ============================================================

N = 30

# Horizontal velocity
u = np.ones((N, N))

# Vertical velocity
v = np.zeros((N, N))

# Create a grid of positions
x, y = np.meshgrid(np.arange(N), np.arange(N))

# Draw velocity arrows
plt.figure(figsize=(8, 8))
plt.quiver(x, y, u, v)

plt.title("Velocity Field")
plt.xlabel("x position")
plt.ylabel("y position")
plt.show()


# ============================================================
# 8. VELOCITY FIELD AROUND AN OBSTACLE
# ============================================================

N = 30

x, y = np.meshgrid(np.arange(N), np.arange(N))

# Start with flow to the right
u = np.ones((N, N))
v = np.zeros((N, N))

# Position of the obstacle
cx = 15
cy = 15
radius = 5

# Distance from the obstacle
distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

# Make the flow turn around the obstacle
v = 3 * (y - cy) / (distance + 1)

# Stop the arrows inside the obstacle
obstacle = distance < radius
u[obstacle] = 0
v[obstacle] = 0

# Draw velocity field
plt.figure(figsize=(8, 8))
plt.quiver(x, y, u, v)

# Draw the obstacle
circle = plt.Circle((cx, cy), radius, fill=False)
plt.gca().add_patch(circle)

plt.xlim(0, N)
plt.ylim(0, N)
plt.gca().set_aspect("equal")

plt.title("Velocity Field Around an Obstacle")
plt.show()


# ============================================================
# 9. PRESSURE FIELD
# ============================================================

N = 30

# Create a simple pressure field
pressure = np.exp(
    -((x - 15) ** 2 + (y - 15) ** 2) / 30
)

plt.figure(figsize=(8, 6))
plt.imshow(pressure, origin="lower")

plt.colorbar(label="Pressure")
plt.title("Pressure Field")
plt.xlabel("x position")
plt.ylabel("y position")

plt.show()


# ============================================================
# 10. EFFECT OF VISCOSITY
# ============================================================

N = 50

# Velocity in the horizontal direction
u = np.zeros((N, N))

# Make the middle region move quickly
u[20:30, :] = 5

viscosity = 0.1

# Neighboring values
a = np.roll(u, 1, axis=0)
b = np.roll(u, -1, axis=0)
c = np.roll(u, 1, axis=1)
d = np.roll(u, -1, axis=1)

# Simple diffusion-like viscosity step
u = u + viscosity * (a + b + c + d - 4 * u)

plt.imshow(u, origin="lower")
plt.colorbar(label="Velocity")
plt.title("Effect of Viscosity")
plt.show()


# ============================================================
# 11. PRESSURE PRODUCING FLUID MOTION
# ============================================================

N = 50

# Pressure is higher on the left
pressure = np.zeros((N, N))
pressure[:, 0:10] = 10

# Calculate pressure gradient in the x direction
pressure_gradient = np.gradient(pressure, axis=1)

# Pressure pushes fluid away from high pressure
velocity = -pressure_gradient

plt.imshow(velocity, origin="lower")
plt.colorbar(label="Velocity caused by pressure")
plt.title("Pressure Producing Fluid Motion")
plt.show()


# ============================================================
# 12. INITIAL FLUID VELOCITY
# ============================================================

N = 50

u = np.zeros((N, N))

# Make the middle region move quickly
u[20:30, :] = 5

plt.imshow(u, origin="lower")
plt.colorbar(label="Velocity")
plt.title("Initial Fluid Velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ============================================================
# 13. VISCOSITY SPREADING FLUID MOTION
# ============================================================

N = 50

u = np.zeros((N, N))

# Fast-moving fluid in the middle
u[20:30, :] = 5

# Viscosity spreads the velocity
u = gaussian_filter(u, sigma=3)

plt.imshow(u, origin="lower")
plt.colorbar(label="Velocity")
plt.title("Viscosity Spreading Fluid Motion")
plt.show()


# ============================================================
# 14. ANIMATED VISCOSITY
# ============================================================

N = 50

u = np.zeros((N, N))

# Fast-moving layer
u[20:30, :] = 5

plt.figure(figsize=(7, 5))

for step in range(20):
    # Spread the velocity through the fluid
    u = gaussian_filter(u, sigma=0.8)

    plt.clf()
    plt.imshow(u, origin="lower")
    plt.colorbar(label="Velocity")
    plt.title(f"Viscosity Spreading Motion - Step {step + 1}")

    plt.pause(0.15)

plt.show()


print("\nProject completed.")
print("This project demonstrates simplified concepts related to fluid dynamics.")
