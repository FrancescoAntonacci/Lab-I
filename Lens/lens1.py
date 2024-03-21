import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''' Funzione per calcolare il seno dell'angolo rifratto in base all'indice di rifrazione'''
def seno_rifratto(sen_incidente, n_lente):
    return sen_incidente / n_lente


def linear_function(x, m, q):
    return m * x + q

''' inseriamo i dati raccolti durante l'esperienza laboratoriale'''
seni_incidenti = np.array([1, 2.3, 3.5, 4.5, 5.4, 6.0, 6.7, 7.3, 7.6, 7.7])
seni_rifratti = np.array([0.8, 1.6, 2.4, 3.1, 3.7, 4.1, 4.5, 4.9, 5.1, 5.2])

'''fittiamo i dati'''
popt, pcov = curve_fit(linear_function, seni_incidenti, seni_rifratti)

plt.figure()
plt.plot(seni_incidenti, seni_rifratti, 'bo', label='Dati sperimentali')
plt.plot(seni_incidenti, linear_function(seni_incidenti, *popt), 'r-', label='Curva di fit')
plt.xlabel('Seno dell\'angolo incidente')
plt.ylabel('Seno dell\'angolo rifratto')
plt.title('Fit dei dati e coefficiente di rifrazione')
plt.legend()
plt.grid(True)

'''eseguiamo anche un grafico per i residui a parte '''
residui = seni_rifratti - linear_function(seni_incidenti, *popt)

plt.figure()
plt.plot(seni_incidenti, residui, 'bo')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Seno dell\'angolo incidente')
plt.ylabel('Residui')
plt.title('Grafico dei residui')
plt.grid(True)

plt.show()

'''calcoliamo l'indice di rifrazione della lente '''
indice_rifrazione_lente = 1 / popt[0]
'''dove popt[0] rappresenta il nostro dato ottimizzato'''
print("L'indice di rifrazione della lente è:", indice_rifrazione_lente)