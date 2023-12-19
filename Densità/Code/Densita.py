import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
##Struttura Dati

# CONVERTIRE! MISURE PRESE IN GRAMMI E MILLIMETRI- FATTO!
## Convenzioni
#1= Alluminio
#2= Acciaio
#3=Ottone
##

def vol_sphere(d,s_d):
    v=4*np.pi*((d/2)**3)/3
    s_v=v*3*s_d/d
    return v,s_v
def vol_parall(a,b,c,s_a,s_b,s_c):
    v=a*b*c
    s_v=(np.sqrt((s_a/a)**2+(s_b/b)**2+(s_c/c)**2))*v
    return v,s_v
def vol_cil(d,h,s_d,s_h):
    v=(np.pi*((d/2)**2)*h)
    s_v=(np.sqrt((s_d*2/d)**2+(s_h/h)**2))*v
    return v,s_v
def vol_ex(da,h,s_da,s_h):
    v=np.tan(np.pi/6)*(1.5)*(da**2)*h
    s_v=v*np.sqrt(((2*(da/s_da)))**2+((s_h/h)**2))
    return v,s_v


def dens(m,v,s_m,s_v):
    ro=(m/v)
    s_ro=np.sqrt((s_m/m)**2+(s_v/v)**2)*ro
    return ro,s_ro


class Sphere():
    def __init__(self,m,s_m,d,s_d,n):
        self.m= m/1000
        self.d=d/1000
        self.n=n
        self.s_m=s_m/1000
        self.s_d=s_d/1000

    def cal_vol(self):
        self.v, self.s_v= vol_sphere(self.d,self.s_d)

    def dens(self):
        self.ro,self.s_ro=dens(self.m,self.v,self.s_m,self.s_v)



class Parall():
    def __init__(self,m,s_m,l,s_l,w,s_w,h,s_h,n):
        self.m=m/1000
        self.l=l/1000
        self.w=w/1000
        self.h=h/1000
        self.n=n
        self.s_m=s_m/1000
        self.s_l=s_l/1000
        self.s_w=s_w/1000
        self.s_h=s_h/1000

    def cal_vol(self):
        self.v,self.s_v= vol_parall(self.l,self.w,self.h,self.s_l,self.s_w,self.s_h)

    def dens(self):
        self.ro,self.s_ro=dens(self.m,self.v,self.s_m,self.s_v)


class cil():
    def __init__(self,m,s_m,d,s_d,h,s_h,n):
        self.m= m/1000
        self.d=d/1000
        self.h=h/1000
        self.n=n
        self.s_m=s_m/1000
        self.s_d=s_d/1000
        self.s_h=s_h/1000

    def cal_vol(self):
        self.v,self.s_v= vol_cil(self.d,self.h,self.s_d,self.s_h)

    def dens(self):
        self.ro,self.s_ro=dens(self.m,self.v,self.s_m,self.s_v)

class ex():
    def __init__(self,m,s_m,d,s_d,h,s_h,n):
        self.m= m/1000
        self.d=d/1000
        self.h=h/1000
        self.n=n
        self.s_m=s_m/1000
        self.s_d=s_d/1000
        self.s_h=s_h/1000

    def cal_vol(self):
        self.v,self.s_v= vol_cil(self.d,self.h,self.s_d,self.s_h)

    def dens(self):
        self.ro,self.s_ro=dens(self.m,self.v,self.s_m,self.s_v)

## Dati


alsolido4=Parall(4.853,0.001,10.05,0.01,10.04,0.01,18.43,0.01,1)
alsolido5=Parall(7.766,0.001,8.14,0.01,17.78,0.01,20.07,0.01,1)
alsolido6=cil(5.789,0.001,11.96,0.01,19.08,0.01,1)
alsolido7=cil(15.777,0.001,18.91,0.01,19.80,0.01,1)
otsolido8=cil(10.764,0.001,9.96,0.01,16.41,0.01,3)
otsolido9=ex(16.474,0.001,9.94,0.01,22.82,0.01,3)
otsolido10=Parall(34.725,0.001,9.98,0.01,9.95,0.01,41.50,0.02,3)
otsolido11=cil(24.528,0.001,9.96,0.01,37.30,0.02,3)
acsolido12=Sphere(3.524,0.001,9.52,0.01,2)
acsolido13=Sphere(8.359,0.001,12.69,0.01,2)
acsolido14=Sphere(11.889,0.001,14.27,0.01,2)
acsolido15=Sphere(28.193,0.001,19.03,0.01,2)
acsolido16=Sphere(44.821,0.001,22.20,0.01,2)

alsolido4.cal_vol()
alsolido5.cal_vol()
alsolido6.cal_vol()
alsolido7.cal_vol()
otsolido8.cal_vol()
otsolido9.cal_vol()
otsolido10.cal_vol()
otsolido11.cal_vol()
acsolido12.cal_vol()
acsolido13.cal_vol()
acsolido14.cal_vol()
acsolido15.cal_vol()
acsolido16.cal_vol()


al_masse=[alsolido4.m,alsolido5.m,alsolido6.m,alsolido7.m]
al_s_masse=[alsolido4.s_m,alsolido5.s_m,alsolido6.s_m,alsolido7.s_m]
al_vol=[alsolido4.v,alsolido5.v,alsolido6.v,alsolido7.v]
al_s_vol=[alsolido4.s_v,alsolido5.s_v,alsolido6.s_v,alsolido7.s_v]

ac_masse=[acsolido12.m,acsolido13.m,acsolido14.m,acsolido15.m,acsolido16.m]
ac_s_masse=[acsolido12.s_m,acsolido13.s_m,acsolido14.s_m,acsolido15.s_m,acsolido16.s_m]
ac_vol=[acsolido12.v,acsolido13.v,acsolido14.v,acsolido15.v,acsolido16.v]
ac_s_vol=[acsolido12.s_v,acsolido13.s_v,acsolido14.s_v,acsolido15.s_v,acsolido16.s_v]

ot_masse=[otsolido8.m,otsolido10.m,otsolido11.m]
ot_s_masse=[otsolido8.s_m,otsolido10.s_m,otsolido11.s_m]
ot_vol=[otsolido8.v,otsolido10.v,otsolido11.v]
ot_s_vol=[otsolido8.s_v,otsolido10.s_v,otsolido11.s_v]
## Funzione dell'algoritmo di Best- Fit
def line(x, m, q):
    return m * x + q

def power_law(x, norm, index):
    return norm * (x**index)


## Algoritmo di Best Fit-Retta Alluminio
popt_al, pcov_al = curve_fit(line, al_masse, al_vol)
m_hat_al, q_hat_al = popt_al
sigma_m_al, sigma_q_al = np.sqrt(pcov_al.diagonal())
ro_al=1/m_hat_al
s_ro_al=(sigma_m_al/(m_hat_al))*ro_al

popt_ac, pcov_ac = curve_fit(line, ac_masse, ac_vol)
m_hat_ac, q_hat_ac = popt_ac
sigma_m_ac, sigma_q_ac = np.sqrt(pcov_ac.diagonal())
ro_ac=1/m_hat_ac
s_ro_ac=(sigma_m_ac/(m_hat_ac))*ro_ac

popt_ot, pcov_ot = curve_fit(line, ot_masse, ot_vol)
m_hat_ot, q_hat_ot = popt_ot
sigma_m_ot, sigma_q_ot = np.sqrt(pcov_ot.diagonal())
ro_ot=1/m_hat_ot
s_ro_ot=(sigma_m_ot/(m_hat_ot))*ro_ot

print("Densità alluminio=",ro_al,"Errore di misura densità alluminio=",s_ro_al,"\nQ_hat e relativo errore", q_hat_al, sigma_q_al,"m_hat_al e s_m_al=",m_hat_al,sigma_m_al )
print("Densità acciaio=",ro_ac,"Errore di misura densità acciaio=",s_ro_ac,"\nQ_hat e relativo errore", q_hat_ac, sigma_q_ac,"m_hat_ac e s_m_ac=",m_hat_ac,sigma_m_ac)
print(r"Densità ottone=",ro_ot,"Errore di misura densità ottone=",s_ro_ot,"\nQ_hat e relativo errore", q_hat_ot, sigma_q_ot,"m_hat_ot e s_m_ot=",m_hat_ot,sigma_m_ot)

## Algoritmo di Best Fit-Legge di potenza
r_spheres=[acsolido12.d/2,acsolido13.d/2,acsolido14.d/2,acsolido15.d/2,acsolido16.d/2]
s_r_spheres=[acsolido12.s_d/2,acsolido13.s_d/2,acsolido14.s_d/2,acsolido15.s_d/2,acsolido16.s_d/2]
m_spheres=[acsolido12.m,acsolido13.m,acsolido14.m,acsolido15.m,acsolido16.m]
s_m_spheres=[acsolido12.s_m,acsolido13.s_m,acsolido14.s_m,acsolido15.s_m,acsolido16.s_m]


pot_popt, pot_pcov = curve_fit(power_law,m_spheres,r_spheres)
norm_hat, index_hat = pot_popt
sigma_norm, sigma_index = np.sqrt(pot_pcov.diagonal())
print("coefficiente[m^3/kg]=",norm_hat,"coefficiente[m^3/kg]=", sigma_norm, "Esponente=",index_hat, "Sigma_esponente=",sigma_index)
###         GRAFICI
###         GRAFICI
###         GRAFICI
##Grafico massa-raggio
plt.figure("Grafico massa-raggio")
plt.errorbar( m_spheres,r_spheres,s_r_spheres,s_m_spheres,  fmt=".")

x = np.linspace(min(m_spheres), max(m_spheres), 100)
plt.plot(x, power_law(x, norm_hat, index_hat))
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Raggio [m]")
plt.xlabel("Massa [Kg]")
plt.grid(which="both", ls="dashed", color="gray")




## Grafico

plt.figure("Grafico massa-volume Alluminio")

x = np.linspace(min(al_masse), max(al_masse), 1000)

plt.errorbar(al_masse, al_vol, al_s_vol, al_s_masse,fmt=".") #Plot dati
plt.plot(x, line(x, m_hat_al, q_hat_al)) #plot Fit

plt.ylabel("Volume [m$^3$]")
plt.xlabel("Massa [kg]")
plt.grid(which="both", ls="dashed", color="gray")

plt.figure("Grafico massa-volume Acciaio")

x = np.linspace(min(ac_masse), max(ac_masse), 1000)

plt.errorbar(ac_masse, ac_vol, ac_s_vol, ac_s_masse,fmt=".")
plt.plot(x, line(x, m_hat_ac, q_hat_ac))

plt.ylabel("Volume [m$^3$]")
plt.xlabel("Massa [kg]")
plt.grid(which="both", ls="dashed", color="gray")


plt.figure("Grafico massa-volume Ottone")

x = np.linspace(min(ot_masse), max(ot_masse), 1000)

plt.errorbar(ot_masse, ot_vol, ot_s_vol, ot_s_masse,fmt=".")
plt.plot(x, line(x, m_hat_ot, q_hat_ot))

plt.ylabel("Volume [m$^3$]")
plt.xlabel("Massa [kg]")
plt.grid(which="both", ls="dashed", color="gray")



## Grafici dei Residui



plt.figure("Grafico dei residui dei solidi di alluminio")
Ral=al_vol-line(np.array(al_masse),m_hat_al,q_hat_al)
plt.errorbar(al_masse,Ral,np.array(al_s_vol),fmt=".") #Scarto ValoriMedi-Fit
plt.grid(ls="dashed", which="both", color="gray")
plt.xlabel("Massa[Kg]")
plt.ylabel("Differenza previsione del fit-misure del volume[m$^3$]")



plt.figure("Grafico dei residui dei solidi di acciaio")
Rac=ac_vol-line(np.array(ac_masse),m_hat_ac,q_hat_ac)
plt.errorbar(ac_masse,Rac,ac_s_vol,fmt=".",label=" ") #Scarto ValoriMedi-Fit
plt.grid(ls="dashed", which="both", color="gray")
plt.xlabel("Massa[Kg]")
plt.ylabel("Differenza previsione del fit-misure del volume[m$^3$]")



plt.figure("Grafico dei residui dei solidi di ottone")
Rot=ot_vol-line(np.array(ot_masse),m_hat_ot,q_hat_ot)
plt.errorbar(ot_masse,Rot,ot_s_vol,fmt=".",label=" ") #Scarto ValoriMedi-Fit
plt.grid(ls="dashed", which="both", color="gray")
plt.xlabel("Massa[Kg]")
plt.ylabel("Differenza previsione del fit-misure del volume[m$^3$]")

##Grafico unificato

plt.figure("Grafico unificato")

x1 = np.linspace(min(al_masse), max(al_masse), 1000)
x2 = np.linspace(min(ac_masse), max(ac_masse), 1000)
x3 = np.linspace(min(ot_masse), max(ot_masse), 1000)

plt.errorbar(al_masse, al_vol, al_s_vol, al_s_masse,fmt=".",label="Masse di alluminio") #Plot dati
plt.plot(x1, line(x1, m_hat_al, q_hat_al)) #plot Fit

plt.errorbar(ac_masse, ac_vol, ac_s_vol, ac_s_masse,fmt=".",label="Masse di acciaio")
plt.plot(x2, line(x2, m_hat_ac, q_hat_ac))

plt.errorbar(ot_masse, ot_vol, ot_s_vol, ot_s_masse,fmt=".",label="Masse di ottone")
plt.plot(x3, line(x3, m_hat_ot, q_hat_ot))

plt.grid(which="both", ls="dashed", color="gray")

plt.ylabel("Volume [m$^3$]")
plt.xlabel("Massa [kg]")
plt.legend()

plt.show()

# plt.savefig(’massa_raggio.pdf’)
