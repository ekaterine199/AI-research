import torch
import matplotlib.pyplot as plt
from model import PendulumNet
from physics import PendulumPhysics
from trainer import PINNTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_physics = torch.linspace(0, 10, 500).view(-1, 1).to(device)
    t_init = torch.tensor([[0.0]], device=device)
    theta_init = torch.tensor([[1.0]], device=device) 

    model = PendulumNet().to(device)
    physics = PendulumPhysics(mu=0.5)
    trainer = PINNTrainer(model, physics)

    print(f"Training on {device}...")
    for epoch in range(5001):
        loss = trainer.train_step(t_physics, t_init, theta_init)
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    model.eval()
    t_test = torch.linspace(0, 10, 100).view(-1, 1).to(device)
    with torch.no_grad():
        theta_pred = model(t_test).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(t_test.cpu().numpy(), theta_pred, label="PINN (Neural Physics)", color='blue', linewidth=2)
    plt.title("Pendulum Motion Learned by PINN")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()