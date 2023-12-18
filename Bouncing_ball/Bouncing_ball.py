import wave

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
## Data
file_path = r'C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Bouncing_ball\Bouncingball_9.wav'
stream = wave.open(file_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)



if stream.getnchannels() == 2: #really important if the acquisition system is a stereo: one input shall be suppressed.
    signal = signal[::2]

t = np.arange(len(signal)) / stream.getframerate()

h_measured=1.00
s_h_measured=0.01
g=9.80513

## Manipulation
a_signal=np.abs(signal)

t_bounces=[0]

for i in range(3,np.size(a_signal)):
    if(np.size(t_bounces)==0):
        if(a_signal[i]/10000>1):
            t_bounces.append(t[i])
            print("Caricamento",i*100/np.size(a_signal))
    elif(max(t_bounces)+0.1-t[i]<0):
        if(a_signal[i]/10000>1):
            t_bounces.append(t[i])
            print("Caricamento",i*100/np.size(a_signal))
t_bounces=t_bounces[:np.size(t_bounces)-1]
s_t_bounces=np.full_like(t_bounces, 0.001)


dt = np.diff(np.array(t_bounces))
s_t_dt=np.full_like(dt,0.001)
# Creazione dell"array con gli indici dei rimbalzi.
n = np.arange(len(dt)) + 1.
# Calcolo dell’altezza massima e propagazione degli errori.
h = g * (dt**2.) / 8.0
print("h=",h)
s_h = 2* np.sqrt(2) * h * s_t_dt / dt

##Function
def expo(n, h0, gamma):
    """Modello di fit.
    """
    return h0 * gamma**n
##Best-fit Algorythm
popt, pcov = curve_fit(expo, n, h, sigma=s_h)
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pcov.diagonal())
print(h0_hat, sigma_h0, gamma_hat, sigma_gamma)

## Plot signal

plt.figure("Audio")
plt.plot(t, signal)
plt.xlabel("Tempo [s]")
#plt.savefig("audio_rimbalzi.pdf")

##Plot
plt.figure("Altezza dei rimbalzi")
# print("hello")
plt.errorbar(n, h, s_h, fmt=".")

x = np.linspace(min(n),max(n), 5000)
plt.plot(x, expo(x, h0_hat, gamma_hat))
plt.yscale("log")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Rimbalzo")
plt.ylabel("Altezza massima [m]")
# plt.savefig("altezza_rimbalzi.pdf")

plt.show()

