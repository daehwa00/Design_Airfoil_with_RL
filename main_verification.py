import os
import json
from NACA import *
from utils import bezier_curve
import numpy as np
from matplotlib import pyplot as plt
from simulation import run_simulation
from OPENFOAM_MAKER import make_block_mesh_dict
import matplotlib.cm as cm
import math

# Load NACA 0012, 4412 data
naca0012 = naca0012
naca4412 = naca4412

data_0012 = data_0012
data_4412 = data_4412

num_points = 36

# Calculate Bezier curves for NACA 0012
bezier_naca0012 = bezier_curve(
    np.array([naca0012["x"], naca0012["y"]]).T, num=num_points
)

# Calculate Bezier curves for NACA 4412
bezier_naca4412 = bezier_curve(
    np.array([naca4412["x"], naca4412["y"]]).T, num=num_points
)


# Save simulation results to a file
def save_simulation_results(filename, results):
    with open(filename, "w") as f:
        json.dump(results, f)


# Load simulation results from a file
def load_simulation_results(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None


# Function to plot airfoil shape
def plot_airfoil(naca, title, ax, color):
    ax.fill(naca[:, 0], naca[:, 1], color=color, alpha=0.3)
    ax.plot(naca[:, 0], naca[:, 1], label=f"{title} Surface", color=color)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title} Airfoil")
    ax.legend()
    ax.grid(True)


# Function to plot CL data
def plot_cl(data, title, ax_cl, colormap):
    colors = cm.get_cmap(colormap, len(data["RN"]))
    for i, rn in enumerate(data["RN"]):
        if rn in data["CL"]:
            ax_cl.plot(
                data["AOA"],
                data["CL"][rn],
                label=f"Re = {rn:.1e}",
                color=colors(i),
                linestyle="--",
            )

    ax_cl.set_title(f"{title} CL vs AOA")
    ax_cl.set_xlabel("AOA (degrees)")
    ax_cl.set_ylabel("CL")
    ax_cl.legend()
    ax_cl.grid(True)


# Function to plot CL/CD data
def plot_cl_cd_ratio(data, title, ax, colormap):
    colors = cm.get_cmap(colormap, len(data["RN"]))
    for i, rn in enumerate(data["RN"]):
        if rn in data["CL"] and rn in data["CD"]:
            cl = np.array(data["CL"][rn], dtype=np.float64)
            cd = np.array(data["CD"][rn], dtype=np.float64)
            # Remove None values
            valid_indices = np.logical_and(cl != None, cd != None)
            cl = cl[valid_indices]
            cd = cd[valid_indices]
            if len(cl) > 0 and len(cd) > 0:
                cl_cd = np.divide(cl, cd, out=np.zeros_like(cl), where=cd != 0)
                cl_cd = np.where(np.abs(cl_cd) > 200, np.nan, cl_cd)  # Remove outliers
                ax.plot(
                    data["AOA"][: len(cl_cd)],
                    cl_cd,
                    label=f"Re = {rn:.1e}",
                    color=colors(i),
                    linestyle="--",
                )

    ax.set_title(f"{title} CL/CD vs AOA")
    ax.set_xlabel("AOA (degrees)")
    ax.set_ylabel("CL/CD")
    ax.legend()
    ax.grid(True)


# Plot NACA 0012 and NACA 4412 in separate subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Plot NACA 0012 Airfoil Shape
plot_airfoil(bezier_naca0012, "NACA 0012", axes[0, 0], "blue")

# Plot NACA 4412 Airfoil Shape
plot_airfoil(bezier_naca4412, "NACA 4412", axes[0, 1], "red")

# Plot CL data for NACA 0012
plot_cl(data_0012, "NACA 0012", axes[1, 0], "viridis")

# Plot CL data for NACA 4412
plot_cl(data_4412, "NACA 4412", axes[1, 1], "plasma")

# Plot CL/CD ratio data for NACA 0012
plot_cl_cd_ratio(data_0012, "NACA 0012", axes[2, 0], "viridis")

# Plot CL/CD ratio data for NACA 4412
plot_cl_cd_ratio(data_4412, "NACA 4412", axes[2, 1], "plasma")

plt.tight_layout()
plt.show()

# Re = V * L / nu
# -> V = Re * nu / L

nu = 1.5e-5  # Kinematic viscosity of air (m^2/s)
RN_0012 = data_0012["RN"]
RN_4412 = data_4412["RN"]

AOA_0012 = data_0012["AOA"]
AOA_4412 = data_4412["AOA"]

simulation_results_0012_file = "simulation/simulation_results_0012.json"
simulation_results_4412_file = "simulation/simulation_results_4412.json"

# Load or run simulations for NACA 0012
simulation_results_0012 = load_simulation_results(simulation_results_0012_file)
if not simulation_results_0012:
    simulation_results_0012 = {rn: {"CL": [], "CD": [], "AOA": []} for rn in RN_0012}
    for rn in RN_0012:
        for aoa in AOA_0012:
            freestream_velocity = rn * nu / 1.0
            make_block_mesh_dict(
                naca0012["x"],
                naca0012["y"],
                aoa=aoa,
                freestream_velocity=freestream_velocity,
            )
            Cd, Cl = run_simulation(verbose=False)
            print(
                f"Re = {rn:.1e}, AOA = {aoa}, CL = {Cl}, CD = {Cd}, freestream_velocity = {freestream_velocity:.2f}m/s"
            )

            simulation_results_0012[rn]["CL"].append(Cl)
            simulation_results_0012[rn]["CD"].append(Cd)
            simulation_results_0012[rn]["AOA"].append(aoa)
    save_simulation_results(simulation_results_0012_file, simulation_results_0012)

# Load or run simulations for NACA 4412
simulation_results_4412 = load_simulation_results(simulation_results_4412_file)
if not simulation_results_4412:
    simulation_results_4412 = {rn: {"CL": [], "CD": [], "AOA": []} for rn in RN_4412}
    for rn in RN_4412:
        for aoa in AOA_4412:
            freestream_velocity = rn * nu / 1.0
            make_block_mesh_dict(
                naca4412["x"],
                naca4412["y"],
                aoa=aoa,
                freestream_velocity=freestream_velocity,
            )
            Cd, Cl = run_simulation(verbose=False)
            print(
                f"Re = {rn:.1e}, AOA = {aoa}, CL = {Cl}, CD = {Cd}, freestream_velocity = {freestream_velocity:.2f}m/s"
            )

            simulation_results_4412[rn]["CL"].append(Cl)
            simulation_results_4412[rn]["CD"].append(Cd)
            simulation_results_4412[rn]["AOA"].append(aoa)
    save_simulation_results(simulation_results_4412_file, simulation_results_4412)


# Function to plot simulation vs experimental CL data
def plot_simulation_vs_experiment_cl(data, simulation_data, title, colormap):
    colors = cm.get_cmap(colormap, len(data["RN"]))
    plt.figure(figsize=(10, 7))
    for i, rn in enumerate(data["RN"]):
        if rn in data["CL"] and str(rn) in simulation_data:
            plt.plot(
                data["AOA"],
                data["CL"][rn],
                label=f"Experimental Re = {rn:.1e}",
                color=colors(i),
                linestyle="--",
            )
            plt.scatter(
                simulation_data[str(rn)]["AOA"],
                simulation_data[str(rn)]["CL"],
                color=colors(i),
                marker="x",
                label=f"Simulation Re = {rn:.1e}",
            )
    plt.title(title)
    plt.xlabel("AOA (degrees)")
    plt.ylabel("CL")
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to plot simulation vs experimental CL/CD ratio data
def plot_simulation_vs_experiment_cl_cd_ratio(data, simulation_data, title, colormap):
    colors = cm.get_cmap(colormap, len(data["RN"]))
    plt.figure(figsize=(10, 7))
    for i, rn in enumerate(data["RN"]):
        if rn in data["CL"] and rn in data["CD"] and str(rn) in simulation_data:
            experimental_cl_cd_ratio = [
                cl / cd if cl is not None and cd is not None and cd != 0 else None
                for cl, cd in zip(data["CL"][rn], data["CD"][rn])
            ]
            experimental_cl_cd_ratio = [
                ratio if ratio is None or np.abs(ratio) <= 200 else None
                for ratio in experimental_cl_cd_ratio
            ]  # Remove outliers
            simulation_cl_cd_ratio = [
                cl / cd if cl is not None and cd is not None and cd != 0 else None
                for cl, cd in zip(
                    simulation_data[str(rn)]["CL"], simulation_data[str(rn)]["CD"]
                )
            ]
            simulation_cl_cd_ratio = [
                ratio if ratio is None or np.abs(ratio) <= 200 else None
                for ratio in simulation_cl_cd_ratio
            ]  # Remove outliers
            plt.plot(
                data["AOA"],
                experimental_cl_cd_ratio,
                label=f"Experimental Re = {rn:.1e}",
                color=colors(i),
                linestyle="--",
            )
            plt.scatter(
                simulation_data[str(rn)]["AOA"],
                simulation_cl_cd_ratio,
                color=colors(i),
                marker="x",
                label=f"Simulation Re = {rn:.1e}",
            )
    plt.title(title)
    plt.xlabel("AOA (degrees)")
    plt.ylabel("CL/CD")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot CL for NACA 0012
plot_simulation_vs_experiment_cl(
    data_0012, simulation_results_0012, "NACA 0012 CL vs AOA", "viridis"
)

# Plot CL/CD for NACA 0012
plot_simulation_vs_experiment_cl_cd_ratio(
    data_0012, simulation_results_0012, "NACA 0012 CL/CD vs AOA", "viridis"
)

# Plot CL for NACA 4412
plot_simulation_vs_experiment_cl(
    data_4412, simulation_results_4412, "NACA 4412 CL vs AOA", "plasma"
)

# Plot CL/CD for NACA 4412
plot_simulation_vs_experiment_cl_cd_ratio(
    data_4412, simulation_results_4412, "NACA 4412 CL/CD vs AOA", "plasma"
)
