import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--test_data_size', type=int, default=1000)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--nn', default='INODE')
parser.add_argument('--nrun', type=int, default=1)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--tf', type=int, default=10)
parser.add_argument('--tf_test', type=int, default=10)
parser.add_argument('--flag', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([0., 0.]).to(device)

t = torch.linspace(0., args.tf, args.data_size).to(device)
t_test = torch.linspace(0., args.tf_test, args.test_data_size).to(device)


class rc_circuit(nn.Module):
    def forward(self, t, y):
        Vin = 1.0 # Input voltage
        R1 = 1.0  # Resistance 1
        R2 = 2.0  # Resistance 2
        C1 = 0.1  # Capacitance 1
        C2 = 0.2  # Capacitance 2

        V1, V2 = y
        dV1dt = (Vin - V1) / (R1 * C1)
        dV2dt = (V1 - V2) / (R2 * C2)
        return torch.tensor([dV1dt, dV2dt])

with torch.no_grad():
    true_y = odeint(rc_circuit(), true_y0, t, method='dopri5')
    test_y = odeint(rc_circuit(), true_y0, t_test, method='dopri5')


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



if __name__ == '__main__':
    start = time.perf_counter()

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=1e-4)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        loss = nn.MSELoss()(pred_y, true_y)
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0 or itr == 1:
            with torch.no_grad():
                print('Iter {:04d} | MSE Loss {:.6f}'.format(itr, loss.item()))

                end = time.time()
        
                if itr == args.niters:
                    elapsed = (time.perf_counter() - start)
                    pred_y_test = odeint(func, true_y0, t_test)
                    mse_t = nn.MSELoss()(pred_y_test, test_y)
                    print('MSE Loss {:.6f}'.format(mse_t.item()))
                    plt.plot(t_test.detach().cpu().numpy(), test_y.detach().cpu().numpy(), linestyle='dashed', label='real')
                    plt.plot(t_test.detach().cpu().numpy(), pred_y_test.detach().cpu().numpy(), label='predicted')
                    plt.xlabel("Time")
                    plt.ylabel("Population")
                    plt.legend()
                    plt.show()
                    #plt.savefig("2capacitorRCTime.pdf")
                    #torch.save(func, lotkaVolterraVanillaNeuralODE.pt)
                                        
