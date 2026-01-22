# 🕸️ Physics-Informed Neural Network (PINN) for Pendulum Dynamics

This repository implements a **Physics-Informed Neural Network (PINN)** to simulate the motion of a damped pendulum. Unlike standard neural networks, PINNs integrate physical laws directly into the loss function, allowing the model to learn from governing equations rather than just raw data.

## 🚀 Overview

The system solves the second-order non-linear Ordinary Differential Equation (ODE) for a pendulum with damping:

$$\frac{d^2\theta}{dt^2} + \mu \frac{d\theta}{dt} + \frac{g}{L} \sin(\theta) = 0$$

## 🛠️ Design Patterns

- **Decoupled Architecture**: Separates the neural network (`model.py`), the physical constraints (`physics.py`), and the training engine (`trainer.py`).
- **Automatic Differentiation**: Leverages PyTorch's `autograd` to compute exact derivatives for the physics loss.
- **Hardware Agnostic**: Supports both CPU and CUDA-enabled GPUs.

## 📦 Installation

```bash
pip install torch numpy matplotlib
```

## 🎮 Execution
Run the simulation:
python main.py

