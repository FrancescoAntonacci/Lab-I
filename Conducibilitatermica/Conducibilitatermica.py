import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
## Generatore di potenza

tensione=10.1
sigma_tensione=0.1
corrente=1.69
sigma_corrente=0.01
potenza=corrente*tensione/2
sigma_potenza=((sigma_tensione/tensione)+(sigma_corrente/corrente))*potenza
##Temperature Alluminio


path0 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/cond_231123/alluminio/'
Tav = []
for i in range(1,16):
    i=str(i)
    path = path0 +  'a' + i +'.txt'
    a,b,c, Ta = np.loadtxt(path, unpack = True, skiprows=4)
    Tav.append(Ta)


Temperature_alluminio=[]
for i in range (0,15):
    Temperature_alluminio.append(np.mean(Tav[i]))

sigma_Temperature_alluminio=[]
for i in range (0,15):
    aus=[]
    aus=Tav[i]
    aus_1=[]
    aus_1=aus[(np.size(aus)-15):]

    aus_2=(max(aus_1)-min(aus_1))
    # print(aus_2)
    sigma_Temperature_alluminio.append(aus_2)

print("Sigma temperature alluminio=",sigma_Temperature_alluminio)



##Dimesioni alluminio Alluminio

path1 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/distanzafori_alluminio.txt'

d_alluminio=np.loadtxt(path1, unpack = True)
diametro_fori_alluminio=0.0041
sigma_d_alluminio=np.full(d_alluminio.shape, (diametro_fori_alluminio/2))

diam_alluminio=0.025
sezione_alluminio=np.pi*diam_alluminio**2/4#Sezione barra

##Temperature Rame


path2 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/cond_231123/rame/'
Trv = []
for i in range(1,21):
    i=str(i)
    path = path2 +  'r' + i +'.txt'
    a1,b1,c1, Tr = np.loadtxt(path, unpack = True, skiprows=4)
    Trv.append(Tr)
# print(Trv)

Temperature_rame=[]
for i in range (0,20):
    Temperature_rame.append(np.mean(Trv[i]))



sigma_Temperature_rame=[]
for i in range (0,20):
    aus=[]
    aus=Trv[i]
    aus_1=[]
    aus_1=aus[(np.size(aus)-15):]
    aus_2=(max(aus_1)-min(aus_1))
    sigma_Temperature_rame.append(aus_2)


print("Sigma temperature rame=",sigma_Temperature_rame)

#
# print(Temperature_rame,np.size(Temperature_rame),sigma_Temperature_rame,np.size(sigma_Temperature_rame))
##Dimesioni alluminio Rame

path3 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/distanzafori_rame.txt'

d_rame=np.loadtxt(path3, unpack = True)

diam_rame=0.025

diametro_fori_rame=0.0041
sigma_d_rame=np.full(d_rame.shape, (diametro_fori_rame/2))

sezione_rame=np.pi*(diam_rame**2)*0.25#Sezione barra
#
# print(d_rame,np.size(d_rame),sigma_d_rame,np.size(sigma_d_rame))

##Funzione Fit
def line(x, m, q):

    return m * x + q
##Fit Alluminio
popt_alluminio, pcov_alluminio = curve_fit(line, d_alluminio,Temperature_alluminio,sigma=sigma_Temperature_alluminio)
m_hat_alluminio, q_hat_alluminio = popt_alluminio
sigma_m_alluminio, sigma_q_alluminio = np.sqrt(pcov_alluminio.diagonal())
# print(m_hat_alluminio, sigma_m_alluminio, q_hat_alluminio, sigma_q_alluminio)
print("m alluminio=",m_hat_alluminio)
##Fit Rame
popt_rame, pcov_rame = curve_fit(line, d_rame,Temperature_rame,sigma=sigma_Temperature_rame)
m_hat_rame, q_hat_rame = popt_rame
sigma_m_rame, sigma_q_rame = np.sqrt(pcov_rame.diagonal())
# print(m_hat_rame, sigma_m_rame, q_hat_rame, sigma_q_rame)
print("m rame=",m_hat_rame)
## Lambda Alluminio e Rame
lambda_alluminio=(potenza/(m_hat_alluminio*sezione_alluminio))
lambda_rame=(potenza/(m_hat_rame*sezione_rame))

print("Lambda alluminio=",lambda_alluminio,"\nLambda rame=",lambda_rame)


## Grafico Alluminio
fig, (al1, al2) = plt.subplots(2)


al1.errorbar(d_alluminio, Temperature_alluminio,sigma_Temperature_alluminio, sigma_d_alluminio,fmt=".",label="Dati misurati") #Grafico dati



x = np.linspace(min(d_alluminio), max(d_alluminio), 1000)
al1.plot(x, line(x, m_hat_alluminio, q_hat_alluminio),label="Fit")# Grafico Fit

al1.grid(which="both", ls="dashed", color="gray")
al1.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')
al1.legend()

##Grafico Alluminio residui

Ra=Temperature_alluminio-line(d_alluminio,m_hat_alluminio,q_hat_alluminio)#Residui dell'alluminio


al2.errorbar(d_alluminio,Ra,sigma_Temperature_alluminio,fmt=".",label="Grafico dei residui") #Scarto ValoriMedi-Fit

al2.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')
al2.grid(ls="dashed", which="both", color="gray")

al2.legend()


## Grafico Rame

fig, (ar1, ar2) = plt.subplots(2)


ar1.errorbar(d_rame, Temperature_rame,sigma_Temperature_rame, sigma_d_rame,fmt=".",label="Dati misurati") #Grafico dati



x = np.linspace(min(d_rame), max(d_rame), 1000)
ar1.plot(x, line(x, m_hat_rame, q_hat_rame),label="Fit")# Grafico Fit

ar1.grid(which="both", ls="dashed", color="gray")
ar1.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')
ar1.legend()

##Grafico rame residui

Rr=Temperature_rame-line(d_rame,m_hat_rame,q_hat_rame)#Residui dell'rame


ar2.errorbar(d_rame,Rr,sigma_Temperature_rame,fmt=".",label="Grafico dei residui") #Scarto ValoriMedi-Fit

ar2.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')
ar2.grid(ls="dashed", which="both", color="gray")
ar2.legend()

plt.show()
