Here's the updated README with additional information about installing OpenFOAM and environment simulation:

---

# 3D Propeller Design Project with Reinforcement Learning

This repository contains the code and resources for a project aimed at optimizing the design of 3D propellers using Reinforcement Learning (RL). The project employs a CFD-based reward system to enhance the aerodynamic performance of propeller designs.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Environment Simulation](#environment-simulation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to design an optimized 3D propeller using reinforcement learning techniques. Traditional mathematical approaches struggle with the complexity and nonlinearity of airfoil design. By leveraging RL, the project efficiently explores a vast design space, automates the optimization process, and provides real-time feedback for continuous improvement.

## Features
- **Reinforcement Learning Model**: Utilizes PPO (Proximal Policy Optimization) for training.
- **CFD Simulation**: Integrates OpenFOAM for aerodynamic simulations.
- **Signed Distance Function (SDF)**: Enhances the input representation for the RL model.
- **Automated Optimization**: Automates the design and evaluation process of propellers.
- **Multi-Objective Optimization**: Considers multiple performance metrics such as lift, drag, and structural stability.

## Installation
To set up the project, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/daehwa00/3D-propeller-Design.git
    cd 3D-propeller-Design
    ```
2. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Install [OpenFOAM](https://openfoam.org/download/):
    - Follow the instructions specific to your operating system to install OpenFOAM.

## Usage
To train the RL model and optimize propeller designs, use the following commands:
- **Train the Model**:
    ```bash
    python train.py
    ```
- **Run Simulations**:
    ```bash
    python simulation.py
    ```
- **Evaluate Results**:
    ```bash
    python main_verification.py
    ```

## Directory Structure
- `NACA/`: Contains NACA airfoil data.
- `OPENFOAM_MAKER/`: Scripts for creating OpenFOAM simulations.
- `model/`: Pre-trained models and training scripts.
- `results/`: Output results from simulations and optimizations.
- `simulation/`: Scripts to run and validate simulations.
- `AirfoilEnv.py`: Custom environment for airfoil optimization.
- `train.py`: Main script for training the RL model.
- `simulation.py`: Runs the simulation environment.
- `utils.py`: Utility functions for data processing and analysis.

## Environment Simulation
The environment simulation is set up to validate the aerodynamic performance of the designed propellers. The simulations are conducted using OpenFOAM, which provides a robust framework for computational fluid dynamics (CFD) analysis. Key validation parameters include:
- **Reynolds Number (Re)**: 170000
- **Dynamic Viscosity (\( \mu \))**: \( 1.5 m^2/s \)
- **Flow Velocity (U)**: 2.55 m/s

Validation results include comparisons of lift coefficients (Cl), drag coefficients (Cd), and moment coefficients (Cm) across various angles of attack (AoA).

## Results
The project successfully demonstrates the capability of reinforcement learning to optimize 3D propeller designs. Key results include:
- Improved lift-to-drag ratio.
- Efficient exploration of the design space.
- Convergence to optimal designs without manual intervention.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to further customize and expand this README as needed.
