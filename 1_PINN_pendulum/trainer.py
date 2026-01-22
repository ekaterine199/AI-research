import torch

class PINNTrainer:
    def __init__(self, model, physics, lr=1e-3):
        self.model = model
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, t_physics, t_init, theta_init):
        self.optimizer.zero_grad()
        
        theta_0_pred = self.model(t_init)
        loss_boundary = torch.mean((theta_0_pred - theta_init)**2)
        
        loss_phys = self.physics.compute_loss(self.model, t_physics)
        
        total_loss = loss_boundary + loss_phys
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()