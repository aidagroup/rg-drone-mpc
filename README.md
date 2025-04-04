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

You can install and run this project using either traditional pip or uv package manager.

### Using uv (Recommended)

1. Install uv if you don't have it already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/rg_mpc_demo.git
cd rg_mpc_demo
```

3. Install dependencies and create a virtual environment:
```bash
uv sync
```
This will create a virtual environment in `.venv` and install all dependencies from the lockfile.

4. Activate the virtual environment (optional):
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

5. Run the simulation:
```bash
uv run python test_mpc.py
```

### Using pip

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

4. Run the simulation:
```bash
python test_mpc.py
```

## Running the Simulation

The simulation will:
1. Initialize the underwater drone and MPC controller
2. Run the simulation for 400 time steps
3. Generate and display an animation of the drone's trajectory
4. Save the animation to "drone_mpc_simulation.mp4"

### Alternative Execution Methods with uv

Run with specific dependencies:
```bash
uv run --with matplotlib --with numpy python test_mpc.py
```

Run a specific function or module:
```bash
uv run -m test_mpc
```

## Docker Integration

To run this project in Docker:

1. Create a Dockerfile:
```dockerfile
FROM python:3.12-slim

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock .python-version ./

# Create virtual environment and install dependencies
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Run the simulation
CMD ["uv", "run", "python", "test_mpc.py"]
```

2. Build and run the container:
```bash
docker build -t underwater-drone-mpc .
docker run -it underwater-drone-mpc
```

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

### uv-related Issues

- If you encounter Python version compatibility issues, check your `.python-version` file
- If dependencies fail to resolve, try running `uv lock` to generate a fresh lockfile
- For more detailed logs, use `uv sync --verbose`

## License

MIT License
