import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2
## Functions



def linear_function(x, m, q):
    return m * x + q

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

def sy_eff(m,x,sx,sy):
    '''
    Questa funzione vuole verificare le ipotesi di best fit di curve fit
    ritorna il vettore delle sigma y efficaci

    '''

    sy_eff=np.sqrt((m*sx)**2+sy**2) # to be adjusted

    return sy_eff

## Other functions

def v(w,l,tc,d):
    '''
    Velocity of  the flag in the lowest potential energy point.
    '''
    v= (w*l)/(tc*d)
    return v


def s_v(w,l,tc,d,s_w,s_l,s_d,s_tc):
    '''
    uncertanty on the velocity of  the flag in the lowest potential energy point.
    '''
    s_v= np.sqrt((l*s_w/(tc*d))**2+(w*s_l/(tc*d))**2+(w*l*s_tc/((tc**2)*d))**2+(w*l*s_d/((d**2)*tc))**2)
    return s_v


def theta(w,l,tc,d,g):
    '''
    theta massimo per oscillazione
    '''
    theta=np.arccos(1-(v(w,l,tc,d)**2)/(2*g*l))

    return theta

def T(theta,l,g):
    '''
    Taylor series expansion of the period given the maximum angle
    '''
    a1=1
    a2=1/16
    a3=0#11/3072
    T=np.pi*2*np.sqrt(l/g)*(a1+a2*theta**2+a3*theta**4)
    return T

def exp(t,v0,lam):
    '''
    Trivial exponential law
    '''
    v=v0*np.e**(lam*t)
    return v



##Dimensions

#Parallelepiped
a=0.281 #lenght
s_a=0.001

b=0.034#thickness
s_b=0.001
c=0.034
s_c=0.001

# # Distance suspension-upper surface
# l0=1.073
# s_l0=0.005#This uncertainty can be manipulated: the measure was quite difficult

# Distance lower surface- flagù


#Dimensions flag
w=0.01980
s_w=0.00005

#  ACCORDING TO OLD MEASURES
# #Distance suspension-CM
# l=l0+b/2
# s_l=np.sqrt(s_l0**2+ (s_b/2)**2)

#Distance suspension-CM
l=1.087
s_l=0.001 #To be adjusted


# Distance CM-flag
dcmf=0.058
s_dcmf=0.001

#Distance suspension-flag

d=l+dcmf
s_d=np.sqrt(s_l**2+s_dcmf**2)


##

file_path=r"C:/Users/zoom3/Documents/Unipi/Laboratorio I/LaboratoryReports/PendulumPeriod/data11042024/p2.txt"

time, period, transit_time=np.loadtxt(file_path,skiprows=4, unpack=True)

s_time_arduino=0.000001# due to arduino specs

s_time=np.full_like(time, s_time_arduino)
s_period=np.full_like(time, s_time_arduino)
s_transit_time=np.full_like(time, s_time_arduino)


## Data acquisition



Mv= w*l/(transit_time*d)
s_Mv= s_v(w,l,transit_time,d,s_w,s_l,s_d,s_transit_time)

theta0=theta(w,l,transit_time,d,9.8)


##Best-fit

popt2,pcov2=curve_fit(T,theta0,period,sigma=s_period,absolute_sigma=True)

popt264

##Plots


plt.figure()

plt.errorbar(theta0,period,s_period,fmt=".")

xx=np.linspace(min(theta0),max(theta0),100000)
yy=T(xx,*popt)
plt.plot(xx,yy)

plt.show()

res1=residuals(exp,popt,Mv,time)

x2_p_value(res1,s_Mv,np.size(popt))


