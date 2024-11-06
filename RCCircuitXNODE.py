import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50,128),
            nn.ELU(),
            nn.Linear(128,50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


model = ODEFunc()
model_path = 'RCCircuitResults/2capacitorRCVanillaNeuralODE.pt'  
model = torch.load(model_path)
true_y0 = torch.tensor([0., 0.]).to(device)

t_pred = torch.linspace(0., 10, 100).to(device)
with torch.no_grad():
    pred_y = odeint(model, true_y0, t_pred, method='rk4').to(device)
    V1, V2 = pred_y[:,0], pred_y[:,1]


V1_grid, V2_grid = torch.meshgrid(torch.linspace(0, max(V1), 10), torch.linspace(0, max(V2), 10))
input_tensor = torch.stack([V1_grid, V2_grid], dim=2).view(-1, 2).to(device)
with torch.no_grad():
    derivatives = model(t_pred, input_tensor)
dV1, dV2 = derivatives[:, 0], derivatives[:, 1]


# Plot the time plot
plt.figure()
plt.plot(t_pred.cpu().numpy(), V1.cpu().numpy(), label='$V_1$', color='green')
plt.plot(t_pred.cpu().numpy(), V2.cpu().numpy(), label='$V_2$', color='red')
plt.xlabel('Time')
plt.ylabel('Voltages')
plt.legend()
plt.grid()
plt.savefig("plots/RCNeuralODETime.pdf")

# Plot the state space plot
plt.figure()
plt.plot(V1, V2, label="$V_1$ vs. $V_2$", color='blue')
plt.xlabel("$V_1$ Voltage")
plt.ylabel("$V_2$ Voltage")
plt.legend()
plt.grid()
plt.savefig("plots/RCNeuralODEState.pdf")

# Create a vector field plot
plt.figure()
length = torch.sqrt(dV1**2 + dV2**2)
sc = plt.quiver(V1_grid, V2_grid, dV1, dV2, length, angles='xy', scale_units='xy', cmap='viridis')
plt.xlim(0, max(V1))
plt.ylim(0, max(V2))
plt.xlabel("$V_1$ Voltage")
plt.ylabel("$V_2$ Voltage")
plt.grid()
cbar = plt.colorbar(sc)
plt.savefig("plots/RCNeuralODEVector.pdf")





