#cd Documents cd Laboratorio I cd LaboratoryReports cd FallingMass
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Misure dirette---mettete i vostri numeri!
# Qui potete anche leggere i dati da file, usando il metodo np.loadtxt(),
# se lo trovate comodo.
t = np.array([0.0, 0.0333, 0.0666, 0.1, 0.1333, 0.1666, 0.2, 0.2333, 0.2666, 0.3, 0.3333, 0.3666, 0.4, 0.4333, 0.4666, 0.5, 0.5333, 0.5666, 0.6, 0.6333])
h = np.array([198.5, 197.5, 195.5, 193.0, 190.0, 185.0, 179.0, 173.0, 165.0,156.5, 146.5, 134.0, 122.0, 108.0, 93.0, 77.0, 60.0, 42.0, 22.0, 3.0])
sigma_h = np.array([0.29, 0.33, 0.38, 0.43, 0.47, 0.52, 0.56, 0.61, 0.65, 0.70, 0.74,0.79, 0.84, 0.88, 0.93, 0.97, 1.0, 1.1, 1.1, 1.2])
# Conversione in m.
h = h / 100.0
sigma_h = sigma_h / 100.0

def parabola(t, a, v0, h0):
    """
    Modello di fit quadratico.
    """
    return 0.5 * a * t**2.0 + v0 * t + h0

# plt.figure("Legge oraria")

# Grafico dei punti sperimentali


# plt.errorbar(t, h, sigma_h, fmt="o")
popt, pcov = curve_fit(parabola, t, h, sigma=sigma_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)


# Grafico del modello di fit.
x = np.linspace(0.0, np.max(t), 100)

"""plt.plot(x, parabola(x, *popt))
plt.xlabel("Tempo [s]")
plt.ylabel("Altezza [m]")
plt.grid(ls="dashed", which="both", color="gray")"""

diffFitData=np.abs((h-parabola(t,*popt)))

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("")
ax1.plot(x, parabola(x, *popt))
ax2.plot(t,diffFitData,"o")
ax2.plot(t,sigma_h,"o")

ax1.set(xlabel='[s]', ylabel='[m]')
ax1.grid(ls="dashed", which="both", color="gray")

ax2.set(xlabel='[s]', ylabel='[m]')
ax2.grid(ls="dashed", which="both", color="gray")


ax1.errorbar(t, h, sigma_h, fmt="o")
popt, pcov = curve_fit(parabola, t, h, sigma=sigma_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)



#plt.savefig(’legge_oraria.pdf’)

plt.show()