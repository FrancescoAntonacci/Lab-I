import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2
## Functions


def mean_and_sample_sigma(v):
    """
    this function calculates the sample standard deviation
    """
    n=v.size
    m=sum(v)/n
    a=np.full_like(v,m)
    a=v-a
    s=sum(a**2)/(n-1)
    std_dev= np.sqrt(s)
    return m,std_dev


def residuals(func,params,yy,xx):
    """
    yy, xx are data
    params are the parameters of the function func
    """
    res= yy-func(xx,*params)

    return res



def x2_p_value(res,s_y,n_par):

    x2=sum((res/s_y)**2)

    p_value1 = 1 - chi2.cdf(x2,(np.size(res)-n_par))
    p_value2=chi2.cdf(x2, (np.size(res)-n_par))

    p_value=min(p_value1,p_value2)

    print("Chi_square=",round(x2,3),"p_value=",round(p_value,3),"Degrees of freedom=",(np.size(res)-n_par))

    return p_value,x2



## Other functions

def v(w,l,tc,d):
    '''
    Velocity of  the flag in the lowest potential energy point.
    '''
    v= (w*l)/(tc*d)
    return v


def s_v(w,l,tc,dcmf,s_w,s_l,s_dcmf,s_tc):
    '''
    uncertanty on the velocity of  the flag in the lowest potential energy point.
    '''
    s_v= np.sqrt((l*s_w/(tc*(dcmf+l)))**2+(w*dcmf*s_l/(tc*(dcmf+l)**2))**2+(w*l*s_tc/((tc**2)*(dcmf+l)))**2+(w*l*s_dcmf/(((dcmf+l)**2)*tc))**2)##
    return s_v



def theta(w,l,tc,d,g):
    '''
    theta massimo per oscillazione
    '''
    theta=np.arccos(1-(v(w,l,tc,d)**2)/(2*g*l))

    return theta

def T(theta0,k,a1,a2,a3):
    '''
    Taylor series expansion of the period given the maximum angle
    '''
    # a1=1
    # a2=1/16       THIS IS WHAT WE EXPECT
    # a3=11/3072



    T=np.pi*2*np.sqrt(k)*(a1+a2*theta0**2+a3*theta0**4)
    return T

def exp(t,v0,lam):
    '''
    Trivial exponential law
    '''
    v=v0*np.e**(lam*t)
    return v


## Dimensions

#Parallelepiped
a=0.281 #lenght
s_a=0.001

b=0.034#thickness
s_b=0.001
c=0.034
s_c=0.001

#Dimensions flag
w=0.01980
s_w=0.00005

#Distance suspension-CM
l=1.087
s_l=0.001 #To be adjusted


# Distance CM-flag
dcmf=0.058
s_dcmf=0.001

#Distance suspension-flag

d=l+dcmf
s_d=np.sqrt(s_l**2+s_dcmf**2)

##PART 1 maxv-time

##Data from Arduino

file_path1=r"C:/Users/zoom3/Documents/Unipi/Laboratorio I/LaboratoryReports/PendulumPeriod/data11042024/p1.txt"

time, period, transit_time=np.loadtxt(file_path1,skiprows=4, unpack=True)

s_time_arduino=0.000001# due to arduino specs

s_time=np.full_like(time, s_time_arduino)
s_period=np.full_like(time, s_time_arduino)
s_transit_time=np.full_like(time, s_time_arduino)


## Data manipulation


#max velocity
maxv= w*l/(transit_time*d)  #It is observed that considering the circumference is useless
s_maxv= s_v(w,l,transit_time,d,s_w,s_l,s_d,s_transit_time)


## Best fit

popt1,pcov1=curve_fit(exp,time,maxv,p0=(2.5,-1/370),sigma=s_maxv,absolute_sigma=True)

v0,lam=popt1
s_v0,s_lam=np.sqrt(pcov1.diagonal())

tau=-1/lam
s_tau= s_lam/lam**2

print("\n v0=",round(v0,4),"+-",round(s_v0,4))
print("\n lam=",round(lam,6),"+-",round(s_lam,6))
print("\n tau=",round(tau,3),"+-",round(s_tau,3))

# X2 and residual first part

res1=residuals(exp,popt1,maxv,time)
x2_p_value(res1,s_maxv,np.size(popt1))

##Plot exp

plt.figure("Plot exp")


#plotting variables
xx=np.linspace(min(time),max(time),10000)


plt.errorbar(time,maxv,s_maxv,s_time,fmt='.',label='Dati raccolti')
plt.errorbar(xx,exp(xx, v0, lam), label="Previsione modello")


plt.xlabel('$t[s]$')
plt.ylabel('$v_{max}[m/s]$')
plt.legend(fontsize='large')
plt.grid(True)

##Plot res exp
plt.figure("Residuals_exponential_law")


#plotting variables
xx=np.linspace(min(time),max(time),10000)

plt.errorbar(time,res1/s_maxv,fmt='.',label='residui')



plt.xlabel('$t[s]$')
plt.ylabel('$residui/\sigma_V$')
plt.legend(fontsize='large')
plt.grid(True)

plt.show()