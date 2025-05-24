# HRC: Hierarchical Reinforcement Learning via Causality

This repository contains the implementation of **Hierarchical Reinforcement Learning via causality (HRC)**, a framework designed for learning goal-based hierarchical policies in environments with causal structures. The framework integrates **causal discovery, structural causal models (SCMs), and hierarchical reinforcement learning (HRL)** to efficiently solve long horizon tasks.

## Features

- **Causal Discovery**: Automatically identifies causal relationships between variables in the environment using SCMs.
- **Hierarchical Reinforcement Learning**: Learns goal-based hierarchical policies to solve tasks efficiently.
- **Parallel Training**: Supports distributed training using MPI for scalability.
- **2D Minecraft Environment**: A simulated environment for testing and training the framework.

## File Structure

The repository is organized as follows:

```
.
├── HRC_train.py          # Main script for training
├── DQN.py                # DQN algorithm implementation
├── SCM.py                # Structural Causal Model (SCM) definition and training functions
├── HRL.py                # Goal-based hierarchical policy definition and training functions
├── models.py             # Neural network models for agents and SCM
├── mc.py                 # 2D Minecraft environment wrapper
├── utils.py              # Utility functions
├── minecraft/            # Implementation of the 2D Minecraft environment
├── graph-search/         # Implementation of the experiment on synthetic data on ER
├── data-collection/      # Create and store the synthetic dataset
├── minicraft_run/        # Implementation of minicraft game
└── README.md             # Documentation
```

## Installation

### Dependencies

The following libraries are required to run the code:

- Python 3.8+
- PyTorch
- NumPy
- mpi4py
- scikit-learn
- NetworkX
- TensorBoard or tensorboardX

Install dependencies using:

```sh
pip install -r requirements.txt
```

### Environment Setup

Install the required Python packages:

```sh
pip install -r requirements.txt
```

Ensure MPI is installed on your system for parallel training. For example, on Ubuntu:

```sh
sudo apt-get install mpich
```

## Command-Line Arguments

The script supports several command-line arguments for customization. Below are some key arguments:

| Argument       | Description                                             |
|---------------|---------------------------------------------------------|
| `--causal`    | Whether to use causal discovery.                        |
| `--device`    | Device to use for training (`cuda` or `cpu`).           |
| `--model_path`| Path to save models and logs.                           |
 | `--cda`| The causal discovery algorithm used (`reg` or `bengio`) |
   
For a full list of arguments, run:

```sh
python HRC_train.py --help
```

## How It Works

1. **Causal Discovery**  
   The framework uses SCMs to discover causal relationships between variables in the environment. This helps in identifying key variables that influence the task.

2. **Hierarchical Policy Learning**  
   The discovered causal structure is used to define subgoals for hierarchical reinforcement learning. The HRL agent learns policies to achieve these subgoals, which are then combined to solve the overall task.

3. **Parallel Training**  
   The framework leverages MPI to distribute data collection and training across multiple processes, enabling faster training on large-scale tasks.

## 2D Minecraft Environment

The framework includes a **2D Minecraft environment** (`mc.py`) for testing and training. The environment simulates a grid-world where the agent must achieve specific goals by interacting with objects and navigating the world.

## Example Workflow

### Train Task-Specific Policies

```sh
mpiexec -n 4 python -u HRC_train.py --model_path ./models/task --causal True
```

### Evaluate the Model

Use TensorBoard to visualize training progress:

```sh
tensorboard --logdir ./models/task/log
