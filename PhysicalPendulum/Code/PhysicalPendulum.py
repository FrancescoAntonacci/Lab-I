## LIBRERIE
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

## DATI

f_n1 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period1-5.txt'
f_n2 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period2-5.txt'
f_n3 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period3-5.txt'
f_n4 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period4-5.txt'
f_n5 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period5-5.txt'
f_n6 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period6-5.txt'
f_n7 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period7-5.txt'
f_n8 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period8-5.txt'
f_n9 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period9-5.txt'
f_n10 = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\Period10-5.txt'
f_h = r'C:\Users\zoom3\Documents\Laboratorio I\LaboratoryReports\PhysicalPendulum\Data\HolesDistances.txt'

transcript=np.loadtxt(f_n1, dtype=float,  delimiter=',')



tran1_5=np.loadtxt(f_n1, dtype=float,  delimiter=',')#Dati estratti da Notepad
tran2_5=np.loadtxt(f_n2, dtype=float,  delimiter=',')
tran3_5=np.loadtxt(f_n3, dtype=float,  delimiter=',')
tran4_5=np.loadtxt(f_n4, dtype=float,  delimiter=',')
tran5_5=np.loadtxt(f_n5, dtype=float,  delimiter=',')
tran6_5=np.loadtxt(f_n6, dtype=float,  delimiter=',')
tran7_5=np.loadtxt(f_n7, dtype=float,  delimiter=',')
tran8_5=np.loadtxt(f_n8, dtype=float,  delimiter=',')
tran9_5=np.loadtxt(f_n9, dtype=float,  delimiter=',')
tran10_5=np.loadtxt(f_n10, dtype=float,  delimiter=',')

per5_1=np.mean(tran1_5)#Valori Medi dei Dati
per5_2=np.mean(tran2_5)
per5_3=np.mean(tran3_5)
per5_4=np.mean(tran4_5)
per5_5=np.mean(tran5_5)
per5_6=np.mean(tran6_5)
per5_7=np.mean(tran7_5)
per5_8=np.mean(tran8_5)
per5_9=np.mean(tran9_5)
per5_10=np.mean(tran10_5)

size=10

sigma_per5_1=np.sqrt(sum((tran1_5-per5_1)**2)/(size*(size-1)))
sigma_per5_2=np.sqrt(sum((tran2_5-per5_2)**2)/(size*(size-1)))
sigma_per5_3=np.sqrt(sum((tran3_5-per5_3)**2)/(size*(size-1)))
sigma_per5_4=np.sqrt(sum((tran4_5-per5_4)**2)/(size*(size-1)))
sigma_per5_5=np.sqrt(sum((tran5_5-per5_5)**2)/(size*(size-1)))
sigma_per5_6=np.sqrt(sum((tran6_5-per5_6)**2)/(size*(size-1)))
sigma_per5_7=np.sqrt(sum((tran7_5-per5_7)**2)/(size*(size-1)))
sigma_per5_8=np.sqrt(sum((tran8_5-per5_8)**2)/(size*(size-1)))
sigma_per5_9=np.sqrt(sum((tran9_5-per5_9)**2)/(size*(size-1)))
sigma_per5_10=np.sqrt(sum((tran10_5-per5_10)**2)/(size*(size-1)))

v_per5=[per5_1,per5_2,per5_3,per5_4,per5_5,per5_6,per5_7,per5_8,per5_9,per5_10]
v_sigma_per5=[sigma_per5_1,sigma_per5_2,sigma_per5_3,sigma_per5_4,sigma_per5_5,sigma_per5_6,sigma_per5_7,sigma_per5_8,sigma_per5_9,sigma_per5_10]



v_per=np.divide(v_per5,5)
v_sigma_per=np.divide(v_sigma_per5,5)


l=1.05
sigma_l=0.001

d_holes=np.loadtxt(f_h, dtype=float,  delimiter=',')
sigma_d_holes=0.001



##Calcolo Centro di massa

c_m=0.528

distances=np.abs(c_m-d_holes)

## FIT
# Definizione dell’accelerazione di gravita‘.
g = 9.81
def period_model(d, l):
    P=2.0 * np.pi * np.sqrt((l**2.0 / 12.0 + d**2.0) / (g * d))
    return P



popt, pcov = curve_fit(period_model, distances, v_per, sigma=v_sigma_per)#VERO E PROPRIO FIT

# l_hat = popt[0]           ?????
# sigma_l = np.sqrt(pcov[0, 0])
# print(l_hat, sigma_l)


## Scarto ValoriMedi-Fit
diffFitData=np.abs((v_per-period_model(distances, l))) #Scarto ValoriMedi-Fit




##Grafici
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("")

x= np.linspace(0.01, 0.5, 1000) # Grafico del modello di fit.

##Grafico 1
ax1.errorbar(distances, v_per, v_sigma_per, sigma_d_holes, fmt="o",label='Measures')#Grafico-dati misurati

ax1.plot(x, period_model(x, l),label='Fit')# Grafico FIT

ax1.set(xlabel='Distance from the centre of mass[m]', ylabel='Period of oscillation[s]')
ax1.grid(ls="dashed", which="both", color="gray")
leg = ax1.legend()

##Grafico 2
ratio=diffFitData/v_sigma_per

ax2.plot(distances,ratio,"o") #Scarto ValoriMedi-Fit


ax2.set(xlabel='Distances from the centre of mass[m]', ylabel='Distance in error barr  fit-measured periods')
ax2.grid(ls="dashed", which="both", color="gray")





plt.show()


##Salvataggio
#plt.savefig(’massa_raggio.pdf’)


