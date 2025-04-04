# Underwater Drone MPC Simulation

## Project Overview

This repository contains a Model Predictive Control (MPC) implementation for an underwater drone system using the regelum package. The simulation demonstrates a drone navigating through a water tank with an obstacle to reach a target hole at the top of the tank.

## Features

- Custom MPC implementation with obstacle avoidance constraints
- Realistic underwater drone dynamics with drag forces and gravity
- Interactive visualization of drone trajectory and control actions
- Animation capabilities to visualize simulation results

## Prerequisites

- Python 3.12 or later
- Required packages:
  - regelum (≥0.1.0)
  - casadi (≥3.7.0)
  - numpy
  - matplotlib

## Project Structure

- **constants.py**: Physical constants and simulation parameters
- **system.py**: Underwater drone dynamics implementation 
- **objective.py**: MPC objective function creation
- **constraints.py**: Obstacle avoidance constraints implementation
- **visualization.py**: Animation and visualization functions
- **test_mpc.py**: Main simulation script

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rg_mpc_demo.git
cd rg_mpc_demo
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
```

This will install the package along with all dependencies specified in pyproject.toml.

## Running the Simulation

To run the simulation, execute the main script:

```bash
python test_mpc.py
```

This will:
1. Initialize the underwater drone and MPC controller
2. Run the simulation for 400 time steps
3. Generate and display an animation of the drone's trajectory
4. Save the animation to "drone_mpc_simulation.mp4"

## Simulation Components

### MPC Controller

The `MPCContinuousWithConstraints` class extends the regelum `MPCContinuous` class with:
- Custom constraint handling capability
- Error recovery mechanism for solver failures

### Drone Model

The `UnderwaterDroneNode` class:
- Implements realistic underwater drone dynamics
- Considers drag forces, gravity, and thrust limitations
- Detects when the drone successfully reaches the hole

### Visualization

The simulation includes:
- Real-time trajectory visualization
- Velocity plot
- Control input visualization
- Animation capabilities for post-simulation analysis

## Customization

You can modify various aspects of the simulation:

1. Physical parameters in `constants.py` (mass, drag, gravity, etc.)
2. Objective function weights in `objective.py`
3. Obstacle dimensions and positions in `constraints.py`
4. Simulation duration by changing the loop iteration count in `test_mpc.py`

## Troubleshooting

If the animation fails to save:
- Ensure you have ffmpeg installed (`sudo apt-get install ffmpeg` on Ubuntu)
- Check file permissions in your output directory
- Try using a different output format (change filename extension to .gif)

## License

MIT License
