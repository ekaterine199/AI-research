import torch

class PendulumPhysics:
    def __init__(self, mu=0.5, g=9.81, L=1.0):
        self.mu = mu
        self.g = g
        self.L = L

    def compute_loss(self, model, t):
        t.requires_grad = True
        theta = model(t)
        
        d_theta = torch.autograd.grad(
            theta, t, torch.ones_like(theta), create_graph=True
        )[0]
        
        d2_theta = torch.autograd.grad(
            d_theta, t, torch.ones_like(d_theta), create_graph=True
        )[0]
        
        # ODE: d2_theta + mu*d_theta + (g/L)*sin(theta) = 0
        residual = d2_theta + self.mu * d_theta + (self.g / self.L) * torch.sin(theta)
        
        return torch.mean(residual**2)