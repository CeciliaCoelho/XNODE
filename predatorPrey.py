import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(y, t, r, a, s, b):
    N, P = y
    dNdt = r * N - a * N * P
    dPdt = -s * P + b * N * P
    return [dNdt, dPdt]

y0 = [40, 9]  # Initial population of prey (N) and predators (P)
t = np.linspace(0, 50, 100)  # Time points
r = 0.1  # Intrinsic growth rate of prey
a = 0.02  # Predation rate
s = 0.3  # Predator death rate
b = 0.01  # Reproduction rate of predators per prey consumed

# Solve the ODEs
solution = odeint(lotka_volterra, y0, t, args=(r, a, s, b))
N, P = solution[:, 0], solution[:, 1]

# Create a grid of N and P values for the vector plot
N_grid, P_grid = np.meshgrid(np.linspace(0, max(N), 20), np.linspace(0, max(P), 20))
dN, dP = lotka_volterra([N_grid, P_grid], t, r, a, s, b)
length = np.sqrt(dN**2 + dP**2)


# Plot the time plot
plt.figure()
plt.plot(t, N, label='Prey', color='green')
plt.plot(t, P, label='Predator', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.savefig("plots/lotkaVolterraTrueTime.pdf")

# Plot the state space plot
plt.figure()
plt.plot(N, P, label='Prey vs. Predators', color='blue')
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.legend()
plt.grid()
plt.savefig("plots/lotkaVolterraTrueState.pdf")

# Create a phase plot
plt.figure()
sc = plt.quiver(N_grid, P_grid, dN, dP, length, angles='xy', scale_units='xy', scale=1 , cmap='viridis')
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.xlim(0, max(N))
plt.ylim(0, max(P))
plt.grid()

cbar = plt.colorbar(sc)
plt.savefig("plots/lotkaVolterraTrueVector.pdf")

# Create a vector field plot 
plt.streamplot(N_grid, P_grid, dN, dP, color='blue', arrowsize=2 )
plt.xlabel('Prey Population (N)')
plt.ylabel('Predator Population (P)')
plt.xlim(0, max(N))
plt.ylim(0, max(P))
plt.grid()

