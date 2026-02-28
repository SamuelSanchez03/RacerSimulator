# RacerSimulator

A 2D racing simulation environment designed for evaluating and training control algorithms and artificial intelligence agents.

## Features

* **Customizable Simulation Environment:** Load custom tracks directly from image files.
* **Basic Vehicle Physics:** Implementation of acceleration, braking, friction, and steering mechanics.
* **Ray-casting Observation System:** The vehicle perceives the environment through multiple distance sensors via ray-casting.
* **Real-time Rendering:** Visualizes the agent's progress using Tkinter.
* **Modular Architecture:** Abstract class design to easily integrate new agents and tracks.

## Technologies

* **Language:** Python.
* **Core Libraries:** NumPy (for mathematical operations), Pillow (for track image processing), and Tkinter (for rendering the graphical interface).

## Project Structure

```text
RacerSimulator/
├── agents/
│   └── random_agent.py      # Random action agent implementation
├── race_sim/
│   ├── agent.py             # Abstract base class for agents
│   ├── env.py               # Simulation environment logic
│   ├── track.py             # Image processing and track bounds logic
│   ├── types.py             # Data types definition (Action, CarState)
│   └── viewer.py            # Tkinter rendering engine
├── scripts/
│   └── watch.py             # Main script to run and watch the simulation
├── tracks/
│   ├── Track0.png           # Track 0 image
│   ├── Track1.png           # Track 1 image
│   └── tracks.xcf           # Source file for tracks
├── requirements.txt         # Project dependencies
└── .gitignore
```

## Installation

It is highly recommended to use a virtual environment to isolate the project dependencies.

### Using venv (Standard Python)
```bash
# Clone the repository
git clone <repository-url>
cd RacerSimulator

# Create the virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate # On Windows
source .venv/bin/activate # On macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Using Anaconda / Miniconda

```bash
# Create the environment with Anaconda
conda create --name racersimulator python=3.11
conda activate racersimulator

# Install dependencies
pip install -r requirements.txt
```

### Installing PyTorch
Although the core project relies on NumPy, integrating Deep Learning agents will require PyTorch. Below are the installation steps depending on your operating system and hardware:

* **Windows / Linux with GPU (NVIDIA - CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

* **Windows / Linux CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

* **macOS (Apple Silicon or Intel):**
```bash
pip install torch torchvision torchaudio
```

> Note: *For specific CUDA versions, please refer to the official PyTorch documentation.*

## Usage

To observe a basic demonstration of the simulator using a random agent, run the watch script:

```bash
python -m scripts.watch
```

### Main Architecture Details

#### Classes and Types (`race_sim/types.py`)
* `Action`: Data class defining vehicle commands.
    * Parameters: `steer` (float, -1.0 to 1.0), `throttle` (float, 0.0 to 1.0), `brake` (float, 0.0 to 1.0).
* `CarState`: Data class storing the current physical state of the vehicle.
    * Parameters: `x` (float), `y` (float), `theta` (float), `velocity` (float).

#### `RaceEnv` Class (`race_sim/env.py`)
The core class that manages physics and simulator interactions.
* `__init__(self, track_path: str, verbose: bool = False)`: Initializes the environment by loading the specified track.
* `reset(self) -> npt.NDArray[np.float64]`: Resets the vehicle's position and returns the initial observation.
* `get_obs(self) -> npt.NDArray[np.float64]`: Calculates and returns normalized distances from the ray sensors.
* `step(self, action: Action) -> tuple[npt.NDArray[np.float64], float, bool]`: Applies an action, updates the physics, and returns the new observation, the reward, and the completion state.
* `render(self) -> None`: Updates the graphical viewer on the screen.

#### `Track` Class (`race_sim/track.py`)
Processes the track image to establish navigable boundaries.
* `is_on_road(self, x: float, y: float) -> bool`: Evaluates if the coordinates correspond to the valid road space.
* `cast_ray(self, start_x: float, start_y: float, theta: float, max_range: int = 180, step_size: int = 2) -> float`: Shoots a virtual ray and calculates the distance to the edge of the track.

#### `Agent` Class (`race_sim/agent.py`)
Base class for implementing logic controllers.
* `act(self, obs: npt.NDArray[np.float64]) -> Action`: (Abstract) Defines how the agent maps observations to actions.
* `reset(self, seed=None)`: (Abstract) Resets the internal state of the agent.

## Upcoming Rendering Features

To improve the visual fidelity and functionality of the `Viewer` in the simulator, the following elements are pending implementation:

* **User Interface (HUD):** Displaying an on-screen speedometer, steering angle indicator, braking status, and current reward.