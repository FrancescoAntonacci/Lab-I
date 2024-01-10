import wave

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
## Data
file_path = r'C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Bouncing_ball\Bouncingball15.wav'
stream = wave.open(file_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)



if stream.getnchannels() == 2: #really important if the acquisition system is a stereo: one input shall be suppressed.
    signal = signal[::2]

t = np.arange(len(signal)) / stream.getframerate()

h_measured=0.37
s_h_measured=0.01
g=9.80513



## Manipulation
a_signal=np.abs(signal)

t_bounces=[2.03]

for i in range(3,np.size(a_signal)):
    if(np.size(t_bounces)==0):
        if(a_signal[i]/8000>1):
            t_bounces.append(t[i])
            print("Caricamento",i*100/np.size(a_signal))
    elif(max(t_bounces)+0.06-t[i]<0):
        if(a_signal[i]/8000>1):
            t_bounces.append(t[i])
            print("Caricamento",i*100/np.size(a_signal))
t_bounces.pop(0)
t_bounces=t_bounces[:np.size(t_bounces)-1]
s_t_bounces=np.full_like(t_bounces, 0.0015)


dt = np.diff(np.array(t_bounces))
s_t_dt=np.full_like(dt,0.0015*np.sqrt(2))
# Creazione dell"array con gli indici dei rimbalzi.
n = np.arange(len(dt)) + 1.
# Calcolo dell’altezza massima e propagazione degli errori.
h = g * (dt**2.) / 8.0
# h[0]=0.37
print("h [m] =",h)
s_h = 2* np.sqrt(2) * h * s_t_dt / dt


a=np.full_like(t_bounces,8000)
plt.figure()
plt.plot(t,a_signal)
plt.errorbar(t_bounces,a,fmt="o")

##Function
def expo(n, h0, gamma,c):
    """Modello di fit.
    """
    return (h0 * gamma**n)+c
##Best-fit Algorythm
p0=(1.459,0.793,1)
popt, pcov = curve_fit(expo, n, h,p0, sigma=s_h)
h0_hat, gamma_hat,c_hat = popt
sigma_h0, sigma_gamma, sigma_c = np.sqrt(pcov.diagonal())
print("h0_hat [m]=",round(h0_hat,3), "\nsigma_h0[m]=",round(sigma_h0,3), "\ngamma_hat=", round(gamma_hat,3),"\nsigma_gamma=", round(sigma_gamma,3), "\nc_hat[m]=", round(c_hat,3), "\nsigma_c[m]=", round(sigma_c,3))
print("h_measured[m]=",h_measured,"s_h_measured[m]=",s_h_measured)
## Plot signal

# plt.figure("Audio")
# plt.plot(t, signal)
# plt.xlabel("Tempo [s]")
# #plt.savefig("audio_rimbalzi.pdf")

##Plot
plt.figure("Altezza dei rimbalzi")
plt.errorbar(n, h, s_h, fmt=".",label="Altezze Stimate")

x = np.linspace(min(n),max(n), 5000)
plt.plot(x, expo(x, h0_hat, gamma_hat,c_hat),label="Previsione di (4)")
plt.yscale("log")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Rimbalzo")
plt.ylabel("Altezza massima [m]")
plt.legend()
# plt.savefig("altezza_rimbalzi.pdf")


##Residuals plot
plt.figure("Grafico dei residui")


res=h-expo(n,h0_hat,gamma_hat,c_hat)

plt.errorbar(n, res, s_h, fmt=".",label="Residui")

plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Rimbalzo")
plt.ylabel("[m]")
plt.legend()
# plt.savefig("altezza_rimbalzi.pdf")



plt.show()
