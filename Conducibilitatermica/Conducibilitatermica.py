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



sigma_Temperature_alluminio = np.full(np.size(Temperature), 0.5)
##Dimesioni alluminio Alluminio

path1 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/distanzafori_alluminio.txt'

d_alluminio=np.loadtxt(path1, unpack = True)
sigma_d_alluminio=np.full(d_alluminio.shape, 0.001)


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



sigma_Temperature_rame = np.full(np.size(Temperature_rame), 0.5)

#
# print(Temperature_rame,np.size(Temperature_rame),sigma_Temperature_rame,np.size(sigma_Temperature_rame))
##Dimesioni alluminio Rame

path3 = r'C:/Users/zoom3/Documents/Laboratorio I/LaboratoryReports/Conducibilitatermica/distanzafori_rame.txt'

d_rame=np.loadtxt(path3, unpack = True)
sigma_d_rame=np.full(d_rame.shape, 0.001)
diam_rame=0.025


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

##Fit Rame
popt_rame, pcov_rame = curve_fit(line, d_rame,Temperature_rame,sigma=sigma_Temperature_rame)
m_hat_rame, q_hat_rame = popt_rame
sigma_m_rame, sigma_q_rame = np.sqrt(pcov_rame.diagonal())
# print(m_hat_rame, sigma_m_rame, q_hat_rame, sigma_q_rame)

## Lambda Alluminio e Rame
lambda_alluminio=(potenza/(m_hat_alluminio*sezione_alluminio))
lambda_rame=(potenza/(m_hat_rame*sezione_rame))

print("Lambda alluminio=",lambda_alluminio,"\nLambda rame=",lambda_rame)


## Grafico Alluminio
fig, (al1, al2) = plt.subplots(2)


al1.errorbar(d_alluminio, Temperature_alluminio,sigma_Temperature_alluminio, sigma_d_alluminio,fmt=".") #Grafico dati



x = np.linspace(min(d_alluminio), max(d_alluminio), 1000)
al1.plot(x, line(x, m_hat_alluminio, q_hat_alluminio))# Grafico Fit


al1.grid(which="both", ls="dashed", color="gray")
al1.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')


##Grafico Alluminio residui

Ra=Temperature_alluminio-line(d_alluminio,m_hat_alluminio,q_hat_alluminio)#Residui dell'alluminio


al2.plot(d_alluminio,Ra,"o") #Scarto ValoriMedi-Fit


al2.set(xlabel='', ylabel='')
al2.grid(ls="dashed", which="both", color="gray")




## Grafico Rame

fig, (al1, al2) = plt.subplots(2)


al1.errorbar(d_rame, Temperature_rame,sigma_Temperature_rame, sigma_d_rame,fmt=".") #Grafico dati



x = np.linspace(min(d_rame), max(d_rame), 1000)
al1.plot(x, line(x, m_hat_rame, q_hat_rame))# Grafico Fit


al1.grid(which="both", ls="dashed", color="gray")
al1.set(xlabel='Posizione [M]', ylabel='Temperatura [$^\\circ$C]')


##Grafico rame residui

Ra=Temperature_rame-line(d_rame,m_hat_rame,q_hat_rame)#Residui dell'rame


al2.plot(d_rame,Ra,"o") #Scarto ValoriMedi-Fit


al2.set(xlabel='', ylabel='')
al2.grid(ls="dashed", which="both", color="gray")

#plt.savefig("posizione_temperatura.pdf")

# A questo punto potete usare i risultati del fit per stimare la
# conducibilita‘ vera e propria...

plt.show()
