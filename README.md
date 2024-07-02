# Design_Airfoil_with_RL

This repository contains the code and resources for a project aimed at optimizing the design of 3D propellers using Reinforcement Learning (RL). The project employs a CFD-based reward system to enhance the aerodynamic performance of propeller designs.

**This README is not perfect, and even if you follow the Installation and Usage, it will not work correctly. If you have more questions, please contact me to daehwa001210@gmail.com.**

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

## Simple Architecture Diagram
<img width="1334" alt="arch" src="https://github.com/daehwa00/3D-propeller-Design/assets/62493036/2c8e1fd8-2b5b-4536-abd2-1e7c1c54c6c8">

- **State \( s_t \)**: The current airfoil shape.
- **SDF**: The Signed Distance Function applied to the airfoil shape.
- **Agent**: The RL agent that determines the action \( a_t \).
- **Action \( a_t \)**: The agent's output action, modifying the airfoil shape.
- **New State \( s_{t+1} \)**: The resulting airfoil shape after applying the action.
- **CFD**: The computational fluid dynamics simulation to evaluate the new airfoil.
- **Calculate Reward**: The process of computing the reward based on the CFD results (Cl and Cd).

## Environment Validation
<img width="1292" alt="CL" src="https://github.com/daehwa00/3D-propeller-Design/assets/62493036/59a09adb-f758-41ab-8efe-d25808fde296">
<img width="1292" alt="CD" src="https://github.com/daehwa00/3D-propeller-Design/assets/62493036/b44b7d06-eeec-4217-8146-f797b6b86de7">
<img width="1270" alt="CM" src="https://github.com/daehwa00/3D-propeller-Design/assets/62493036/65e9d45b-b705-44d8-bc76-67d0a4fba3e1">

To ensure the reliability of the simulation results, a comparison between experimental data and simulation outcomes was conducted. The validation results showed that while the lift coefficient (Cl) matched well with experimental data, the drag coefficient (Cd) and moment coefficient (Cm) did not align as closely.

These results highlight the need for further research to improve the accuracy of Cd and Cm predictions. If you are interested in collaborating to advance this research, **please feel free to contact me.**

## Environment Simulation
The environment simulation is set up to validate the aerodynamic performance of the designed propellers. The simulations are conducted using OpenFOAM, which provides a robust framework for computational fluid dynamics (CFD) analysis. Key validation parameters include:
- **Reynolds Number (Re)**: 170000
- **Dynamic Viscosity (\( \mu \))**: \( 1.5 m^2/s \)
- **Flow Velocity (U)**: 2.55 m/s

Validation results include comparisons of lift coefficients (Cl), drag coefficients (Cd), and moment coefficients (Cm) across various angles of attack (AoA).

## Agent Action Design

<img width="995" alt="RL_airfoil_action" src="https://github.com/daehwa00/3D-propeller-Design/assets/62493036/bb587070-5508-45d6-908a-aa14e3bfbebc">

The agent's action generates a 2D output \((x, r)\) representing points on a plane. These points are connected to form a Convex Hull, which is then used as the airfoil shape. This method allows the RL model to iteratively improve the airfoil design by adjusting the position and radius of these points.


## Results
The project successfully demonstrates the capability of reinforcement learning to optimize 3D propeller designs. Key results include:
- Improved lift-to-drag ratio.
- Efficient exploration of the design space.
- Convergence to optimal designs without manual intervention.
![airfoil_animation_with_labels (1)](https://github.com/daehwa00/3D-propeller-Design/assets/62493036/0862ad6b-a374-42c0-bec5-8caad147f52f)


This figure continues to show the creation of a high lift-drag airfoil.
In the lower right-hand corner of the figure, we've represented airfoil's lift-drag ratio.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
