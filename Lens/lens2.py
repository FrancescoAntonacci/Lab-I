import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import chi2
## Functions



def linear_function(x, m, q):
    return m * x + q

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

## OMEGA CLASS
class data_set():
    def __init__(self,path,label1,func,s_x,s_y):


        self.x1,self.y1=np.loadtxt(path,unpack=True)

        self.s_x1=np.full_like(self.x1,s_x) # Must be adjusted
        self.s_y1=np.full_like(self.y1,s_y) # Must be adjusted

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


    def plot(self):


            if (np.size(list(self.label1))>0):

                plt.figure(self.label1,figsize=(10, 10))

                plt.errorbar(self.x1,self.y1,self.s_y1,self.s_x1,fmt=".",label=self.label1)

                self.xx1=np.linspace(min(self.x1),max(self.x1),10000)

                plt.plot(self.xx1,self.func(self.xx1,*self.params1),label=self.label1+" previsione modello")
                plt.xlabel('$t[s]$')
                plt.ylabel('$[]$')
                plt.legend(fontsize='large')
                plt.grid(True)
                plt.show()
                # plt.savefig(self.label1)




    def x2_res_p(self):


        if (np.size(list(self.label1))>0):

            plt.figure(self.label1+" residuals",figsize=(10, 10))

            self.res1=residuals(self.func,self.params1,self.y1,self.x1)
            self.dof1=np.size(self.params1)
            self.p_value,self.x21=x2_p_value(self.res1,self.s_y1,self.dof1)

            plt.errorbar(self.x1,self.res1,self.s_y1,fmt=".",label=self.label1+" residuals")

            plt.xlabel('$x[s]$')
            plt.ylabel('$y[]$')
            plt.legend(fontsize='large')
            plt.grid(True)
            plt.show()
            # plt.savefig(self.label1+"_residuals")


##

file_path=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Lens\ndata1.txt"


##

data1=data_set(file_path,"punti",linear_function,1,1)
data1.p01=[1,0]

##
data1.fit()
data1.plot()
data1.x2_res_p()

