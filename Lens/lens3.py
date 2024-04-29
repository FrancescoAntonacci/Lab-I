import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2


## Setting plot params

fontsize=20
params = {
    'figure.figsize': (8*1.618,8),  # Figure size
    'axes.labelsize': fontsize,       # Axis label size
    'axes.titlesize': fontsize,       # Title size
    'xtick.labelsize': fontsize,      # X-axis tick label size
    'ytick.labelsize': fontsize,      # Y-axis tick label size
    'legend.fontsize': fontsize,      # Legend font size
    'lines.linewidth': 2,       # Line width
    'grid.linewidth': 1,        # Grid line width
    'grid.alpha': 0.5,          # Grid transparency
    'savefig.dpi': 600,         # Resolution of saved figures
    'savefig.transparent': True # Save figures with transparent background
}

plt.rcParams.update(params)
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

## OMEGA CLASS
class data_set():
    def __init__(self,path,label1,func,s_x,s_y):


        self.x1,self.y1=np.loadtxt(path,unpack=True)


        self.s_x1=100*np.full_like(self.x1,s_x)/self.x1**2 # Must be adjusted
        # self.s_y1=np.full_like(self.y1,s_y)/self.y1**2 # Must be adjusted
        self.s_y1=100*np.linspace(0.5,s_y,np.size(s_y))/self.y1**2 # Must be adjusted


        self.x1=100/self.x1
        self.y1=100/self.y1



        self.label1=label1

        self.func=func



        self.p01=[]

        self.params1=[]

        self.s_params1=[]


    def fit(self):

        print(self.p01)

        if (np.size(list(self.label1))>0): # For the first item
            self.popt, self.pcov = curve_fit(self.func, self.x1, self.y1,p0=self.p01,sigma=self.s_y1,absolute_sigma=True,maxfev=30000)
            self.params1= self.popt
            self.s_params1= np.sqrt(self.pcov.diagonal())


            for i in range(0,np.size(self.params1)):
                print(round(self.params1[i],3),"+-",round(self.s_params1[i],3))
            self.s_y0=self.s_y1
            for i in range(5):

                self.s_y1=sy_eff(self.params1[0],self.x1,self.s_x1,self.s_y0)
                self.popt, self.pcov = curve_fit(self.func, self.x1, self.y1,sigma=self.s_x1,absolute_sigma=True)
                self.params= self.popt
                self.s_params= np.sqrt(self.pcov.diagonal())

    def plot(self):


            if (np.size(list(self.label1))>0):

                plt.figure(self.label1)

                plt.errorbar(self.x1,self.y1,self.s_y1,self.s_x1,fmt=".",label=self.label1)

                self.xx1=np.linspace(min(self.x1),max(self.x1),10000)

                plt.plot(self.xx1,self.func(self.xx1,*self.params1),label="Previsione best-fit")
                plt.xlabel('$x[m^-1]$')
                plt.ylabel('$y[m^-1]$')
                plt.legend()
                plt.grid(True)
                plt.show()
                # plt.savefig(self.label1)




    def x2_res_p(self):


        if (np.size(list(self.label1))>0):

            plt.figure(self.label1+" Residuilen")

            self.res1=residuals(self.func,self.params1,self.y1,self.x1)
            self.dof1=np.size(self.params1)
            self.p_value,self.x21=x2_p_value(self.res1,self.s_y1,self.dof1)

            plt.errorbar(self.x1,self.res1/self.s_y1,np.full_like(self.x1,1),fmt=".",label="Residui")

            plt.xlabel('$x[m^-1]$')
            plt.ylabel('$Residui/\sigma$')
            plt.legend()
            plt.grid(True)
            plt.show()
            # plt.savefig(self.label1+"_residuals")


##

file_path=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Lens\data3.txt"


##

s_p=0.5/np.sqrt(12)
s_q=1

data1=data_set(file_path,"Dati raccolti",linear_function,s_p,s_q)
data1.p01=[1,0]

##

data1.fit()
data1.plot()
data1.x2_res_p()

print("f=",1/(data1.params[1]),"+-",(data1.s_params[1])/(data1.params[1]**2))
