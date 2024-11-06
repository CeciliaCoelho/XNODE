import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.ELU(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

model = ODEFunc()
model_path = 'lotkaVolterraResults/lotkaVolterraVanillaNeuralODE.pt'  
model = torch.load(model_path)
true_y0 = torch.tensor([40., 9.]).to(device)

t_pred = torch.linspace(0., 50, 100).to(device)
with torch.no_grad():
    pred_y = odeint(model, true_y0, t_pred, method='rk4').to(device)
    prey, predator = pred_y[:,0], pred_y[:,1]



prey_grid, predator_grid = torch.meshgrid(torch.linspace(0, max(prey), 10), torch.linspace(0, max(predator), 10))
input_tensor = torch.stack([prey_grid, predator_grid], dim=2).view(-1, 2).to(device)
with torch.no_grad():
    derivatives = model(t_pred, input_tensor)
dPrey, dPredator = derivatives[:, 0], derivatives[:, 1]


# Plot the time plot
plt.figure()
plt.plot(t_pred.cpu().numpy(), prey.cpu().numpy(), label='Prey', color='green')
plt.plot(t_pred.cpu().numpy(), predator.cpu().numpy(), label='Predator', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.savefig("plots/lotkaVolterraNeuralODETime.pdf")

# Plot the state space plot
plt.figure()
plt.plot(prey, predator, label="Prey vs. Predator", color='blue')
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.legend()
plt.grid()
plt.savefig("plots/lotkaVolterraNeuralODEState.pdf")

# Create a vector field plot
plt.figure()
length = torch.sqrt(dPrey**2 + dPredator**2)
sc = plt.quiver(prey_grid, predator_grid, dPrey, dPredator, length, angles='xy', scale_units='xy', cmap='viridis')
plt.xlim(0, max(prey))
plt.ylim(0, max(predator))
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.grid()
cbar = plt.colorbar(sc)
plt.savefig("plots/lotkaVolterraNeuralODEVector.pdf")





