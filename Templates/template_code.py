import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2
import os

# Specify the directory path you want to change to
directory_path = r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Templates"

# Change the current working directory
os.chdir(directory_path)



## Pimp my plot

fontsize=14
params = {
    'figure.figsize': (5*1.618,10),  # Figure size
    'axes.labelsize': fontsize,       # Axis label size
    'axes.titlesize': fontsize,       # Title size
    'xtick.labelsize': fontsize,      # X-axis tick label size
    'ytick.labelsize': fontsize,      # Y-axis tick label size
    'legend.fontsize': fontsize,      # Legend font size
    'lines.linewidth': 2,       # Line width
    'grid.linewidth': 1,        # Grid line width
    'grid.alpha': 0.5,          # Grid transparency
    'savefig.dpi': 400,         # Resolution of saved figures
    'savefig.transparent': True, # Save figures with transparent background
    'legend.loc': 'upper right' # Legend location

}

plt.rcParams.update(params)

## Functions

def residuals(func,params,yy,xx):
    """
    yy, xx are data
    params are the parameters of the function func
    """
    res= yy-func(xx,*params)

    return res



def x2_p_value(res,s_y,n_par):
    '''
    All you need to calculate x2 and p_value
    '''

    x2=sum((res/s_y)**2)

    p_value1 = 1 - chi2.cdf(x2,(np.size(res)-n_par))
    p_value2=chi2.cdf(x2, (np.size(res)-n_par))

    p_value=min(p_value1,p_value2)

    print("Chi_square=",round(x2,3),"p_value=",round(p_value,3),"Degrees of freedom=",(np.size(res)-n_par))

    return p_value,x2

## Fit law- drop it here!

def k(x,m,q):
    return m*x+q

## OMEGA CLASS
class data_set():
    def __init__(self,path,label,func,s_x,s_y):


        self.x,self.y=np.loadtxt(path,unpack=True)

        self.s_x=np.full_like(self.x,s_x)
        self.s_y=np.full_like(self.y,s_y)

        self.label=label

        self.func=func

        self.p0=[] # Educated guess for best-fit algorithm

    def fit(self):


        self.popt, self.pcov = curve_fit(self.func, self.x, self.y,p0=self.p0,sigma=self.s_y,absolute_sigma=True,maxfev=300)
        self.params= self.popt
        self.s_params= np.sqrt(self.pcov.diagonal())

        print("Params:",self.params)
        print("s_params",self.s_params)


        self.res=residuals(self.func,self.params,self.y,self.x)
        self.dof=np.size(self.params)
        self.p_value,self.x2=x2_p_value(self.res,self.s_y,self.dof)


    def plot(self):


        self.fig, (self.plot_fit,self.plot_res) = plt.subplots(2)

        self.plot_fit.errorbar(self.x,self.y,self.s_y,self.s_x,fmt=".",label=self.label)

        self.xx=np.linspace(min(self.x),max(self.x),10000)
        self.yy=self.func(self.xx,*self.params)
        self.plot_fit.plot(self.xx,self.yy,label="Previsione best-fit")

        self.plot_fit.set(xlabel='$:)$', ylabel='$:)$') #plot_fit labels and legend
        self.plot_fit.legend()
        self.plot_fit.grid(True)

        self.plot_res.errorbar(self.x,self.res/self.s_y,np.full_like(self.x,1),fmt=".",label="Residui")

        self.plot_res.set(xlabel='$:)$', ylabel='$residui/\sigma$') #plot_fit labels and legend
        self.plot_res.legend()
        self.plot_res.grid(True)

        self.fig.savefig(self.label)
        plt.show()


## Give your file path!

file_path=r"deleteme.txt"


## Uncertainties: the exteem is your! This is your duty !
s_x=1
s_y=2

## Call a first data set,.

data1=data_set(file_path,"deleteme", k,s_x,s_y)
data1.p0=[1,2] # give here the initial guess before calling the functions

## Call the functions
data1.fit()
data1.plot()