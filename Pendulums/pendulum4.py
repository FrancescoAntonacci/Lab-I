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


def theta_t(t,theta_0,tau):
    return theta_0*np.e**(-t/tau)

def x_t_pen(t,A,w,phi,c):
    return A*np.cos(w*t+phi)+c

def x_t_fr(t,A,w,phi,c,tau):
    return A*(np.e**(-t/tau))*np.cos(w*t+phi)+c

def phipb():
    phip=(phi1+phi2)/2
    phib=(phi1-phi2)/2

def wpwb_calc(wf,wc):
    wp=(wc+wf)/2
    wb=(wc-wf)/2
    return wp,wb

def x_t_beats(t,A0,wp,wb,phip,phib,c,tau):

    x=(2*A0*(np.e**(-t/tau))*np.cos(wp*t+phip)*np.cos(wb*t+phib))+c

    return x

def A_calc(xx):
    A=(max(xx)-min(xx))/2
    return A

def c_calc(xx):
    c=(max(xx)+min(xx))/2
    return c

def phi_calc(xx,yy,A,c):
    phi=np.arccos((yy[0]-c)/A)
    return phi

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


def p0_calc(xx,yy):

    """
    """
    A=A_calc(yy)
    c=c_calc(yy)
    phi=phi_calc(xx,yy,A,c)

    i=np.argmax(yy)
    y=yy[i]

    t0=xx[i]
    while(yy[i]>=c):
        i=i+1

    t1=xx[i]

    pi_4=t1-t0

    w=np.pi/(2*pi_4)
    v=[A,w,phi,c]
    return v


## Classes : easiest way to ULTRA SPAGHETTI CODE

class data_set():
    def __init__(self,path,label1,label2,func,s_t,s_s):
        self.t1,self.s1,self.t2,self.s2=np.loadtxt(path,unpack=True,skiprows=5)

        self.s_t1=self.s_t2=np.full_like(self.t1,s_t) # Must be adjusted
        self.s_s1=self.s_s2=np.full_like(self.s1,s_s) # Must be adjusted

        self.label1=label1
        self.label2=label2

        self.func=func



        self.p01=[]
        self.p02=[]

        self.params1=[]
        self.params2=[]

        self.s_params1=[]
        self.s_params2=[]


    def fit(self):

        print(self.p01)

        if (np.size(list(self.label1))>0): # For the first item
            self.popt, self.pcov = curve_fit(self.func, self.t1, self.s1,p0=self.p01,sigma=self.s_s1,absolute_sigma=True,maxfev=30000)
            self.params1= self.popt
            self.s_params1= np.sqrt(self.pcov.diagonal())

            print(self.label1,"\n")

            for i in range(0,np.size(self.params1)):
                print(round(self.params1[i],3),"+-",round(self.s_params1[i],3))


        print(self.p02)

        if (np.size(list(self.label2))>0): #For the second item
            self.popt, self.pcov = curve_fit(self.func, self.t2, self.s2,p0=self.p02,sigma=self.s_s2,absolute_sigma=True,maxfev=30000)
            self.params2= self.popt
            self.s_params2= np.sqrt(self.pcov.diagonal())

            print(self.label2,"\n")

            for i in range(0,np.size(self.params2)):
                print(round(self.params2[i],3),"+-",round(self.s_params2[i],3))

    def plot(self):


            if (np.size(list(self.label1))>0):

                plt.figure(self.label1,figsize=(10, 10))

                plt.errorbar(self.t1,self.s1,self.s_s1,self.s_t1,fmt=".",label=self.label1)

                self.xx1=np.linspace(min(self.t1),max(self.t1),10000)

                plt.plot(self.xx1,self.func(self.xx1,*self.params1),label=self.label1+" previsione modello")
                plt.xlabel('$t[s]$')
                plt.ylabel('$[]$')
                plt.legend(fontsize='large')
                plt.grid(True)
                plt.show()
                # plt.savefig(self.label1)


            if (np.size(list(self.label2))>0):

                plt.figure(self.label2,figsize=(10, 10))

                plt.errorbar(self.t2,self.s2,self.s_s2,self.s_t2,fmt=".",label=self.label2)

                self.xx2=np.linspace(min(self.t2),max(self.t2),10000)

                plt.plot(self.xx2,self.func(self.xx2,*self.params2),label=self.label2+" previsione modello")
                plt.xlabel('$t[s]$')
                plt.ylabel('$[]$')
                plt.legend(fontsize='large')
                plt.grid(True)
                plt.show()
                # plt.savefig(self.label2)


    def x2_res_p(self):


        if (np.size(list(self.label1))>0):

            plt.figure(self.label1+" residuals",figsize=(10, 10))

            self.res1=residuals(self.func,self.params1,self.s1,self.t1)
            self.dof1=np.size(self.params1)
            self.p_value,self.x21=x2_p_value(self.res1,self.s_s1,self.dof1)

            plt.errorbar(self.t1,self.res1,self.s_s1,fmt=".",label=self.label1+"residui")

            plt.xlabel('$t[s]$')
            plt.ylabel('$[]$')
            plt.legend(fontsize='large')
            plt.grid(True)
            plt.show()
            # plt.savefig(self.label1+"_residuals")


        if (np.size(list(self.label2))>0):

            plt.figure(self.label2+" residuals",figsize=(10, 10))

            self.res2=residuals(self.func,self.params2,self.s2,self.t2)
            self.dof2=np.size(self.params2)
            self.p_value,self.x22=x2_p_value(self.res2,self.s_s2,self.dof2)

            plt.errorbar(self.t2,self.res2,self.s_s2,fmt=".",label=self.label2+"residui")

            plt.xlabel('$t[s]$')
            plt.ylabel('$[]$')
            plt.legend(fontsize='large')
            plt.grid(True)
            plt.show()
            # plt.savefig(self.label2+"_residuals")
## Data

s_t=0.05 # To be taken in serious consideration!
s_s=1 #!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!


# Data single pendulum 1

f_path1=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\p_1_a5.txt"


# Data pendulum with friction 2

f_path2=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\p_2_a5.txt"

# Data pendulums in phase  3

f_path3=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\p_3_a5_2.txt"


# Data pendulums out of phase 4

f_path4=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\p_4_2.txt"


# Data beats 5

f_path5=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Pendulums\p_5_1.txt"


## p_1

print("\n\nP_1")

p_1=data_set(f_path1,"","Pendolo singolo",x_t_pen,s_t,s_s)


#Adjust initial guesses
p02=p0_calc(p_1.t2,p_1.s2)
p_1.p02=p02

p_1.p02[2]=np.pi/4
p_1.p02[1]=4.5


p_1.fit()
p_1.plot()
p_1.x2_res_p()

## p_2

print("\n\nP_2")
p_2=data_set(f_path2,"","Pendolo singolo smorzato",x_t_fr,s_t,s_s)

p_2.p02=np.append(p_1.params2,[4])

p_2.fit()
p_2.plot()
p_2.x2_res_p()




## p_3
print("\n\nP_3")

p_3=data_set(f_path3,"Pendolo in fase 1","Pendolo in fase 2",x_t_fr,s_t,s_s)


p_3.p01=np.append(p_1.params2,[4])
p_3.p02=np.append(p_1.params2,[4])

p_3.fit()
p_3.plot()
p_3.x2_res_p()

## p_4

print("\n\nP_4")

p_4=data_set(f_path4,"Pendolo in controfase 1","Pendolo in controfase 2",x_t_fr,s_t,s_s)


p_4.p01=np.append(p_1.params2,[4])
p_4.p02=np.append(p_1.params2,[4])

p_4.fit()
p_4.plot()
p_4.x2_res_p()


## p_5

print("\n\nP_5")

p_5=data_set(f_path5,"Battimenti pendolo 1","Battimenti pendolo 2",x_t_beats,s_t,s_s)

p_5.p01=[90,4.4,0.08,-0.01,np.pi/2,490,60]
p_5.p02=[90,4.4,0.07,-0.3,0,490,70]

p_5.fit()
p_5.plot()
p_5.x2_res_p()



## Contacci
"""
I'm not writing a new class just for 2 points!

This part was so bad..
"""
# Googled data
d_brass= 8730
d_aluminum= 2700

# Dimension of the stick

l_sti=0.476
s_l_sti=0.001

b1_sti=0.0084
s_b1_sti=0.00005

b2_sti=0.0084
s_b2_sti=0.00005

# Dimensions of the cilinder
a_cil=0.0124
s_a_cil=0.00005

d_cil=0.069
s_d_cil=0.001

#volumes

v_sti=l_sti*b1_sti*b2_sti
s_v_sti=v_sti*((s_l_sti/l_sti)+(s_b1_sti/b1_sti)+(s_b2_sti/b2_sti))

v_cil=(np.pi/4)*((d_cil)**2)*a_cil
s_v_cil=(np.pi/4)*((s_a_cil/a_cil)+2*(s_d_cil/d_cil))*v_cil


m_sti=d_aluminum*v_sti
s_m_sti=d_aluminum*s_v_sti

m_cil=d_brass*v_cil
s_m_sti=d_aluminum*s_v_cil

# Inertia
I=(1/3)*m_sti*(l_sti**2+b1_sti**2)+0.5*((d_cil/2)**2)*m_cil+((l_sti+d_cil/2)**2)*m_cil #Inertia
s_I="bla,bla,bla"

# Centre of mass with rispect to the pole
xcm=((l_sti/2)*(m_sti)+(l_sti+d_cil/2)*(m_cil))/(m_cil+m_sti)
s_xcm="bla,bla,bla"

# Expected period

g=9.81 #I'm not commenting this
s_g=0.001

#to simplify formulas
N=(m_cil+m_sti)*g*(xcm)
D=I
w0=np.sqrt(N/D)

s_r=s_d_cil/2 #to simplify formulas
r=d_cil/2

der_b1=(1/(2*w0))*( ((l_sti**2*d_aluminum*b1_sti)*D-N*(0.33*d_aluminum*(2*l_sti**3*b1_sti+4*b1_sti**3*l_sti)))/D**2)
der_l=(1/(2*w0))*( ((g*(2*l_sti*d_aluminum*b1_sti**2+d_brass*np.pi*a_cil*r**2))*D-N*(0.33*d_aluminum*(3*l_sti**2*b1_sti**2+b1_sti**4)+2*(l_sti+r)*(np.pi*d_brass*r**2*a_cil)))/D**2)
der_g=w0/(2*g)
der_r=(1/(2*w0))*( ((d_brass*np.pi*r**2*a_cil+(l_sti+r)*d_brass*2*np.pi*r*a_cil)*D-N*(r**3*np.pi*d_brass*a_cil+2*(l_sti+r)*(np.pi*d_brass*r**2*a_cil)+(l_sti+r)**2*2*r*d_brass*a_cil))/D**2)
s_w0=np.sqrt((der_b1*s_b1_sti)**2+(der_l*s_l_sti)**2+(der_g*s_g)**2+(der_r*s_r)**2)

print("w0 Pendolo singolo (Teoria)=",round(w0,3),"+-",round(s_w0,3))