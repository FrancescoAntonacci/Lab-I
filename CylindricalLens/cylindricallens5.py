import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2

## Fit function

def line(x,m,q):
    '''
    Don't be silly
    '''
    return (m*x) +q

def fit_model(pars,x):
    return pars[0]*x+pars[1]

def conjpoint(x,f):
    y=1/(1/f-1/x)
    return y


def refr(r,f,s_r,s_f):
    '''
    this function calculates the refractive index
    '''
    eta=r/(2*f-r)
    s_eta= np.sqrt(((s_r*2*f)/(2*f-r)**2)**2+(s_f*r/(2*f-r)**2)**2)

    print("Contributo a s_eta di s_r",((s_r*2*f)/(2*f-r)**2))
    print("Contributo a s_eta di s_f",(s_f*r/(2*f-r)**2))

    return eta,s_eta

def sy_eff(m,sx,sy):
    '''
    Questa funzione vuole verificare le ipotesi di best fit di curve fit
    ritorna il vettore delle sigma y efficaci

    '''
    sy_eff=np.sqrt((m*sx)**2+sy**2) #sigma y efficace

    return sy_eff


def calculate_p_value(df, chi_square):
    p_value = 1 - chi2.cdf(chi_square, df)
    return p_value

# Example values
dof = 2



##Data


file_path = r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\CylindricalLens\data080320242.txt"

t,s=np.loadtxt(file_path,unpack=True,delimiter=",") #t: elephone coordinates, s: screen coordinates
t=t/1000 #everything in meters
s=s/1000 #everything in meters
s_t=np.full_like(t,1)/1000 #everything in meters
s_s=np.full_like(s,1)/1000 #everything in meters
l=np.full_like(t,800) #lens coordinates
l=l/1000 #everything in meters
s_l=np.full_like(t,1)/1000 #everything in meters

p=abs(t-l)
s_p=np.sqrt(s_t**2)
q=abs(s-l)
A_foc=np.linspace(6,4,np.size(s_s))/1000
s_q=np.sqrt(s_s**2+(A_foc**2)/12)

x=1/p
s_x=s_p/(p**2)
y=1/q
s_y=s_q/(q**2)

cir4=1090/1000 #everything in meters
s_cir4=1/1000 #everything in meters
r=cir4/(8*np.pi)
s_r=s_cir4/(8*np.pi)


##Best Fit-Algorythm

#This ignore x errors

popt, pcov = curve_fit(line, x, y,sigma=s_y,absolute_sigma=True)
m_hat, q_hat= popt
s_m, s_q_hat= np.sqrt(pcov.diagonal())
s_y0=s_y
for i in range(5):
    s_y=sy_eff(m_hat,s_x,s_y0)
    popt, pcov = curve_fit(line, x, y,sigma=s_y,absolute_sigma=True)
    m_hat, q_hat= popt
    s_m, s_q_hat= np.sqrt(pcov.diagonal())


## Focal point

f= 1/q_hat
s_f=s_q_hat/(q_hat**2)


print("m=",round(m_hat,3),"+-",round(s_m,3),"\nq_hat=",round(q_hat,3),"+-",round(s_q_hat,3),"\nf=",round(f,3),"+-",round(s_f,3))

##Residuals computation

res = y - line(x, m_hat, q_hat)

##Chi-Square test and p_value

x2=sum((res/s_y)**2)
p_value=calculate_p_value(dof,x2)

print("Chi_square=",round(x2,3),"p_value=",round(p_value,3),"Degrees of freedom=",2)



##Plots
fig, (p_f, re) = plt.subplots(2)


xx=np.linspace(min(x),max(x),10000)
yy=line(xx,m_hat,q_hat)


p_f.errorbar(x,y,s_y,s_x,fmt=".",label="Dati sperimentali")
p_f.plot(xx,yy,label="Algoritmo di Best-fit ")
p_f.set(xlabel='$x=1/p[m^{-1}]$', ylabel='$y=1/q[m^{-1}]$')
p_f.grid(which="both", ls="dashed", color="gray")
p_f.legend()

##Residuals plots


re.errorbar(x, res, s_y, fmt=".",label="Residui")
re.set(xlabel='$x [m^{-1}]$', ylabel='$[m^{-1}]$')
re.grid(True)
re.legend()

## Useful plot

plt.figure()

pp=np.linspace(max(p),min(p),100)
qq=conjpoint(pp,f)

plt.errorbar(p,q,s_p,s_q,fmt='.',label="Dati sperimentali")
plt.plot(pp,qq,label="Algoritmo di Best_fit ")

plt.legend()

plt.show()



##Eta

eta,s_eta=refr(r,f,s_r,s_f)
print("Eta=",round(eta,3),"+-",round(s_eta,3))


