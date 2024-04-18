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

    print("Chi_square=",round(x2,3),"p_value=",round(p_value,30),"Degrees of freedom=",(np.size(res)-n_par))

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

def s_theta(w,l,d,g,t,s_w,s_l,s_d,s_g,s_t):

    t0=1/np.sqrt(1-(1-((w**2)*l)/((t**2)*(d**2)*2*g))**2)
    t1=((2*w*l*s_w)/((t**2)*(d**2)*2*g))**2
    t2=((2*w**2*s_l)/((t**2)*(d**2)*2*g))**2
    t3=((w**2*l*s_t)/((t**3)*(d**2)*2*g))**2
    t4=((w**2*l*s_d)/((t**2)*(d**3)*2*g))**2
    t5=(((w**2)*l*s_g)/((t**2)*(d**2)*2*(g**2)))**2

    s_theta=t0*np.sqrt(t1+t2+t3+t4+t5)

    return s_theta



def T(theta0,a1,a2,a3):
    '''
    Taylor series expansion of the period given the maximum angle
    '''
    # a1=1
    # a2=1/16       THIS IS WHAT WE EXPECT
    # a3=11/3072



    T=np.pi*2*(a1+a2*theta0**2+a3*theta0**4)
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
s_l=0.001#To be adjusted


# Distance CM-flag
dcmf=0.058
s_dcmf=0.001

#Distance suspension-flag

d=l+dcmf
s_d=np.sqrt(s_l**2+s_dcmf**2)


#
g=9.805
s_g=0.001

##Data from Arduino

file_path1=r"C:/Users/zoom3/Documents/Unipi/Laboratorio I/LaboratoryReports/PendulumPeriod/data11042024/p3.txt"

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




##PART 2 maxtheta0-Period

theta0=theta(w,l,transit_time,d,g)
s_theta=s_theta(w,l,d,g,transit_time,s_w,s_l,s_d,s_g,s_transit_time)



##BEST FIT


s_period=np.sqrt(s_period**2+(2*np.pi*np.sqrt(l/g)*(theta0/8)*s_theta)**2)

popt2,pcov2=curve_fit(T,theta0,period,p0=(0.11 ,0.11/16,0.11*11/3072),sigma=s_period,absolute_sigma=True)

a1,a2,a3=popt2
s_a1,s_a2,s_a3=np.sqrt(pcov2.diagonal())


print("\n a1=",round(a1,6),"+-",round(s_a1,6))
print("\n a2=",round(a2,6),"+-",round(s_a2,6))
print("\n a3=",round(a3,6),"+-",round(s_a3,6))



# X2 and residual first part

res2=residuals(T,popt2,period,theta0)
x2_p_value(res2,s_period,np.size(popt1))


##

plt.figure("Plot Periods-Theta",figsize=(10, 10))


#plotting variables
xx=np.linspace(min(theta0),max(theta0),10000)


plt.errorbar(theta0,period,s_period,s_theta,fmt='.',label='Dati raccolti')
plt.errorbar(xx,(T(xx,a1,a2,a3)), label="Previsione modello con 3 termini")


plt.xlabel('$theta$')
plt.ylabel('$T[s]$')
plt.legend(fontsize='large')
plt.grid(True)

## COME SI GODE QUANDO LAB 1 TORNA

plt.figure("Plot Periods-time",figsize=(10, 10))


#plotting variables
xx=np.linspace(min(time),max(time),10000)

plt.errorbar(time,period,s_period,s_time,fmt='.',label='Dati raccolti')
plt.errorbar(time,(T(theta0,a1,a2,a3)), label="Previsione modello con 3 termini")


plt.xlabel('$t[s]$')
plt.ylabel('$T[s]$')
plt.legend(fontsize='large')
plt.grid(True)
##Plot res exp



plt.figure("Residuals_period_law",figsize=(10, 10))


#plotting variables


plt.errorbar(time,res2/s_period,fmt='.',label='residui con 3 termini')



plt.xlabel('$theta$')
plt.ylabel('$residui/\sigma_T$')
plt.legend(fontsize='large')
plt.grid(True)






##
plt.show()