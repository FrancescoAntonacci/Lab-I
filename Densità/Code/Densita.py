import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
##Struttura Dati

# CONVERTIRE! MISURE PRESE IN GRAMMI E MILLIMETRI

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


def dens(m,v,s_m,s_v):
    ro=(m/v)
    s_ro=((s_m/m)+(s_v/v))*ro
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
        self.h=s/1000
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
        self.h=s/1000
        self.n=n
        self.s_m=s_m/1000
        self.s_d=s_d/1000
        self.s_h=s_h/1000

    def cal_vol(self):
        self.v,self.s_v= vol_cil(self.d,self.h,self.s_d,self.s_h)

    def dens(self):
        self.ro,self.s_ro=dens(self.m,self.v,self.s_m,self.s_v)

## Dati
n_file=16#numero file
solid = r'C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Densita\Dati\solid'

meas=[]
for x in range(4,(n_file+1)):
    x=str(x)
    file_path=solid+x+'.txt'
    aus=np.loadtxt(file_path,dtype=str,max_rows=1)
    if(aus=='sfera'):
        aus1=np.loadtxt(file_path,skiprows=2,delimiter=',')
        sfera=Sphere(aus1[0],aus1[1],aus1[2],aus1[3],aus1[4])
        meas.append(sfera)

    elif(aus=='para'):
        aus1=np.loadtxt(file_path,skiprows=2,delimiter=',')
        para=Parall(aus1[0],aus1[1],aus1[2],aus1[3],aus1[4],aus1[5],aus1[6],aus1[7],aus1[8])
        meas.append(para)
    elif(aus=='cil'):
        aus1=np.loadtxt(file_path,skiprows=2,delimiter=',')
        para=cil(aus1[0],aus1[1],aus1[2],aus1[3],aus1[4],aus1[5],aus1[6],aus1[7],aus1[8])
        meas.append(para)
    elif(aus=='ex'):
        aus1=np.loadtxt(file_path,skiprows=2,delimiter=',')
        para=ex(aus1[0],aus1[1],aus1[2],aus1[3],aus1[4],aus1[5],aus1[6],aus1[7],aus1[8])
        meas.append(para)
    else:
        print("ERRORE!!!!!")

## Funzione dell'algoritmo di Best- Fit
def line(x, m, q):
    return m * x + q

def power_law(x, norm, index):
    return norm * (x**index)

## Algoritmo di Best Fit-Retta
popt, pcov = curve_fit(line, V, m)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)
## Algoritmo di Best Fit-Legge di potenza
popt, pcov = curve_fit(power_law, r, m)
norm_hat, index_hat = popt
sigma_norm, sigma_index = np.sqrt(pcov.diagonal())
print(norm_hat, sigma_norm, index_hat, sigma_index)

## Grafico
x = np.linspace(0., 4000., 1000)
plt.figure("Grafico massa-volume")
plt.errorbar(V, m, sigma_m, sigma_V, fmt=".")



plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Volume [mm$^3$]")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="dashed", color="gray")
plt.savefig("massa_volume.pdf")

plt.figure("Grafico massa-raggio")
plt.errorbar(r, m, sigma_m, sigma_r, fmt=".")

x = np.linspace(4., 10., 100)
plt.plot(x, power_law(x, norm_hat, index_hat))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Raggio [mm]")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="dashed", color="gray")
#plt.savefig(’massa_raggio.pdf’)

plt.show()

##CODICE BALDINI






