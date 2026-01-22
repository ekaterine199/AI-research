import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(), # helps not to get 0 after first derivative
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)
    
def physics_loss(model, t, mu=0.5, g=9.81, L=1.0):
    t.requires_grad = True
    theta = model(t)
    
    # First Derivative
    d_theta = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
    
    # Second derivative
    d2_theta = torch.autograd.grad(d_theta, t, torch.ones_like(d_theta), create_graph=True)[0]
    
    # Equation itself 
    pde_res = d2_theta + mu * d_theta + (g/L) * torch.sin(theta)
    
    return torch.mean(pde_res**2)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

t_physics = torch.linspace(0, 10, 500).view(-1, 1)

t_initial = torch.tensor([[0.0]], requires_grad=True)
theta_initial_target = torch.tensor([[1.0]])

for epoch in range(5000):
    optimizer.zero_grad()
    
    theta_0_pred = model(t_initial)
    loss_boundary = torch.mean((theta_0_pred - theta_initial_target)**2)
    
    loss_phys = physics_loss(model, t_physics)
    
    total_loss = loss_boundary + loss_phys
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")


t_test = torch.linspace(0, 10, 100).view(-1, 1)
theta_pred = model(t_test).detach().numpy()

plt.figure(figsize=(10, 5))
plt.plot(t_test.numpy(), theta_pred, label="PINN Prediction")
plt.title("Pendulum (taught by physics)")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (radians)")
plt.grid(True)
plt.legend()
plt.show()