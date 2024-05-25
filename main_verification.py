from NACA import *
from utils import bezier_curve
import numpy as np
from matplotlib import pyplot as plt

# Load NACA 0012, 4412 data
naca0012 = naca0012
naca4412 = naca4412

num_points = 36

# Calculate Bezier curves for NACA 0012
naca0012 = bezier_curve(np.array([naca0012["x"], naca0012["y"]]).T, num=num_points)

# Calculate Bezier curves for NACA 4412
naca4412 = bezier_curve(np.array([naca4412["x"], naca4412["y"]]).T, num=num_points)

# Plot NACA 0012 and NACA 4412 in separate subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot NACA 0012
axes[0].plot(naca0012[:, 0], naca0012[:, 1], label="Upper Surface")
axes[0].set_aspect("equal")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("NACA 0012 Airfoil")

# Plot NACA 4412
axes[1].plot(naca4412[:, 0], naca4412[:, 1], label="Upper Surface")
axes[1].set_aspect("equal")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_title("NACA 4412 Airfoil")

plt.tight_layout()
plt.show()
