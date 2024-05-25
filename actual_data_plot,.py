# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from NACA import *

# NACA 0012 and 4412 data (Assumed to be loaded correctly as per the user's request)


# Function to plot airfoil shape
def plot_airfoil(naca, title):
    plt.figure(figsize=(10, 6))
    plt.plot(naca["x"], naca["y"], marker="o")
    plt.title(f"{title} Airfoil Shape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# Function to plot CL and CD data
def plot_cl_cd(data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    for rn in data["RN"]:
        if rn in data["CL"]:
            axs[0].plot(data["AOA"], data["CL"][rn], label=f"Re = {rn}")
        if rn in data["CD"]:
            axs[1].plot(data["AOA"], data["CD"][rn], label=f"Re = {rn}")

    axs[0].set_title("Coefficient of Lift (CL) vs Angle of Attack (AOA)")
    axs[0].set_xlabel("AOA (degrees)")
    axs[0].set_ylabel("CL")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title("Coefficient of Drag (CD) vs Angle of Attack (AOA)")
    axs[1].set_xlabel("AOA (degrees)")
    axs[1].set_ylabel("CD")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Plot NACA 0012 Airfoil Shape
plot_airfoil(naca0012, "NACA 0012")

# Plot NACA 4412 Airfoil Shape
plot_airfoil(naca4412, "NACA 4412")

# Plot CL and CD data for NACA 4412
plot_cl_cd(data_4412)
