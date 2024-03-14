import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2

## Introduction
"""
This code wants to represent the ULTIMATE OVERKILL weapon to approach Lab1

"""
## Functions proper to the experiment

def w0_calc(m,g,l,I):
    return np.sqrt(m*g*l/I)


def theta_t(theta_0,t,tau):
    return theta_0*np.e**(-t/tau)

def x_t(t,A0,wf,wc,phi1,phi2):
    wp=(wc+wf)/2
    phip=(phi1+phi2)/2
    wb=(wc-wf)/2
    phib=(phi1-phi2)/2

    x=2*A0*np.cos(wp*t+phip)*np.cos(wb*t+phib)

    return x

## Functions for data manipulation

def para(x,a,b,c):
    return a*x**2+b*x+c

def d_para(x,a,b):
    return 2*a*x+b


def sample_sigma(v):
    """
    this function calculates the sample standard deviation
    """
    n=v.size
    m=sum(v)/n
    a=np.full_like(v,m)
    a=v-a
    s=sum(a**2)/(n-1)
    std_dev= np.sqrt(s)
    return std_dev

def sy_eff(d_func,params,x,sx,sy):
    '''
    Questa funzione vuole verificare le ipotesi di best fit di curve fit
    ritorna il vettore delle sigma y efficaci

    '''

    sy_eff=np.sqrt((d_func(x,params[0],params[1])*sx)**2+sy**2) # to be adjusted

    return sy_eff


def residuals(func,params,yy,xx):
    """
    yy, xx are data
    params are the parameters of the function func
    """
    res= yy-func(xx,*params)

    return res



def x2_p_value(res,s_y,dof):

    x2=sum((res/s_y)**2)

    p_value1 = 1 - chi2.cdf(x2, dof)
    p_value2=chi2.cdf(x2, dof)

    p_value=min(p_value1,p_value2)

    print("Chi_square=",round(x2,3),"p_value=",round(p_value,3),"Degrees of freedom=",dof)
    return p_value,x2




## Classes : easiest way to ULTRA SPAGHETTI CODE

class data_set():
    def __init__(self,path,label1,label2,func):
        self.t1,self.s1,self.t2,self.s2=np.loadtxt(path,unpack=True,delimiter=",")

        self.s_t1=self.s_t2=np.full_like(self.t1,1) # Must be adjusted
        self.s_s1=self.s_s2=np.full_like(self.s1,1) # Must be adjusted

        self.label1=label1
        self.label2=label2

        self.func=func

        self.p01
        self.p02

    def fit(self):

        if (np.size(list(self.label1))>0): # For the first item
            self.popt, self.pcov = curve_fit(self.func, self.t1, self.s1,sigma=self.s_s1,absolute_sigma=True)
            self.params1= self.popt
            self.s_params1= np.sqrt(self.pcov.diagonal())
            self.s_s0= self.s_s1


            for i in range(0,np.size(self.params1)):
                print(round(self.params1[i],3),"+-",round(self.s_params1[i],3))




        # if (np.size(list(self.label2))>0): #For the second item
        #     self.popt, self.pcov = curve_fit(self.func, self.t2, self.s2,sigma=self.s_s2,absolute_sigma=True)
        #     self.params2= self.popt
        #     self.s_params2= np.sqrt(self.pcov.diagonal())
        #     self.s_s0= self.s_s2


            for i in range(0,np.size(self.params2)):
                print(round(self.params2[i],3),"+-",round(self.s_params2[i],3))

    def plot(self):

            plt.figure(self.label1+" and "+self.label2)

            if (np.size(list(self.label1))>0):
                plt.errorbar(self.t1,self.s1,self.s_s1,self.s_t1,fmt=".",label=self.label1)

                self.xx1=np.linspace(min(self.t1),max(self.t1),10000)

                plt.plot(self.xx1,self.func(self.xx1,*self.params1),label=self.label1+" previsione modello")

            # if (np.size(list(self.label2))>0):
            #     plt.errorbar(self.t2,self.s2,self.s_s2,self.s_t2,fmt=".",label=self.label2)
            #
            #     self.xx2=np.linspace(min(self.t2),max(self.t2),10000)
            #
            #     plt.plot(self.xx2,self.func(self.xx2,*self.params2),label=self.label2+" previsione modello")

            plt.xlabel('$t[s]$')
            plt.ylabel('$[]$')
            plt.legend()
            plt.grid(True)
            plt.show()

    def x2_res_p(self):
        plt.figure(self.label1+" residuals")

        if (np.size(list(self.label1))>0):
            self.res1=residuals(self.func,self.params1,self.s1,self.t1)
            self.dof1=np.size(self.params1)
            self.p_value,self.x21=x2_p_value(self.res1,self.s_s1,self.dof1)

            plt.errorbar(self.t1,self.res1,self.s_s1,fmt=".",label=self.label1+"residuals")

            plt.xlabel('$t[s]$')
            plt.ylabel('$[]$')
            plt.legend()
            plt.grid(True)
            plt.show()

        plt.figure(self.label1+" residuals")

        if (np.size(list(self.label2))>0):
            self.res2=residuals(self.func,self.params2,self.s2,self.t2)
            self.dof2=np.size(self.params2)
            self.p_value,self.x22=x2_p_value(self.res2,self.s_s2,self.dof2)

            plt.errorbar(self.t2,self.res2,self.s_s2,fmt=".",label=self.label2+"residuals")

            plt.xlabel('$t[s]$')
            plt.ylabel('$[]$')
            plt.legend()
            plt.grid(True)
            plt.show()
## Data

# Data single pendulum 1

f_path1=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\test1.txt"

ds1=data_set(f_path1,"si","pure",para)

# Data pendulums in phase  2

f_path2=r""


# Data pendulums out of phase 3

f_path3=r""

# Data beats 4

f_path4=r""


##
            # for i in range(5):        ##THIS PART OF CODE NEEDS A LOT OF MAINTENANCE: it would be more effective if it were processed out of the class
            #
            #     self.s_s1=sy_eff(self.d_func,self.params1,self.t1,self.s_t1,self.s_s1)
            #     self.popt, self.pcov = curve_fit(self.func, self.t1, self.s1,sigma=self.s_s1,absolute_sigma=True)
            #     self.params= self.popt
            #     self.s_params= np.sqrt(self.pcov.diagonal())

##


## Plots


