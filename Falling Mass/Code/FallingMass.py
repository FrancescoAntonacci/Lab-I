## LIBRERIE
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

## DATI
file_name = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\FallingMass\FallingMassDistances.txt'#CAMBIARE DIPENDENTEMENTE DAL DISPOSITIVO

with open(file_name, 'r') as f:
    lines = f.read()

transcript=np.loadtxt(file_name, dtype=float,  delimiter=',')

nummeas=int(transcript.size*0.5)#numero di misure

tin=1#Tempo Finale in secondi
tfi=4#Tempo Iniziale in secondi
t=np.linspace(tin,tfi,nummeas)#Array dei tempi

h=transcript[0::2]
sigma_h=transcript[1::2]



## FIT
def parabola(t, a, v0, h0):

    """
    Modello di fit quadratico.
    """
    return 0.5 * a * t**2.0 + v0 * t + h0

popt, pcov = curve_fit(parabola, t, h, sigma=sigma_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))

## Scarto ValoriMedi-Fit
diffFitData=np.abs((h-parabola(t,*popt))) #Scarto ValoriMedi-Fit


##Grafici
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("")

x = np.linspace(np.min(t), np.max(t), 100) # Grafico del modello di fit.

##Grafico 1
ax1.errorbar(t, h, sigma_h, fmt="o",label='Measures')#Dati misurati

ax1.plot(x, parabola(x, *popt),label='Fit')#FIT

ax1.set(xlabel='Time[s]', ylabel='Distance[pixels]')
ax1.grid(ls="dashed", which="both", color="gray")
leg = ax1.legend()

##Grafico 2
ratio=diffFitData/sigma_h

ax2.plot(t,ratio,"o") #Scarto ValoriMedi-Fit


ax2.set(xlabel='time[s]', ylabel='Distance in error bars  fit-measured distances')
ax2.grid(ls="dashed", which="both", color="gray")


##Salvataggio
#plt.savefig("FallingMassPlot.pdf")

plt.show()
