## LIBRERIE
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

## DATI
file_name = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\FallingMass\FallingMass\FallingMassDistances.txt'#CAMBIARE DIPENDENTEMENTE DAL DISPOSITIVO

with open(file_name, 'r') as f:
    lines = f.read()

print(lines)

transcript=np.loadtxt(file_name, dtype=float,  delimiter=',')
print(transcript)

nummeas=int(transcript.size*0.5)#numero di misure
print(nummeas)
tin=0#Tempo Finale in secondi
tfi=1#Tempo Iniziale in secondi
t=np.linspace(tin,tfi,nummeas)#Array dei tempi

h=transcript[0::2]
sigma_h=transcript[1::2]
print(h)
print(sigma_h)

##CONVERSIONI
h = h / 100.0
sigma_h = sigma_h / 100.0


## FIT
def parabola(t, a, v0, h0):

    """
    Modello di fit quadratico.
    """
    return 0.5 * a * t**2.0 + v0 * t + h0

popt, pcov = curve_fit(parabola, t, h, sigma=sigma_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)

## Scarto ValoriMedi-Fit
diffFitData=np.abs((h-parabola(t,*popt))) #Scarto ValoriMedi-Fit


##Grafici
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("")

x = np.linspace(0.0, np.max(t), 100) # Grafico del modello di fit.

##Grafico 1
ax1.errorbar(t, h, sigma_h, fmt="o",label='Measures')#Dati misurati

ax1.plot(x, parabola(x, *popt),label='Fit')#FIT

ax1.set(xlabel='Time[s]', ylabel='Distance[m]')
ax1.grid(ls="dashed", which="both", color="gray")
leg = ax1.legend()

##Grafico 2
ax2.plot(t,diffFitData,"o",label='Difference central tendency-Fit prediction ')#Scarto ValoriMedi-Fit

ax2.plot(t,sigma_h,"o",label='Standard deviation of the measures')#Errori dalle misure

ax2.set(xlabel='[s]', ylabel='[m]')
ax2.grid(ls="dashed", which="both", color="gray")
leg = ax2.legend()
##Salvataggio
plt.savefig(’FallingMassPlot.pdf’)

plt.show()
