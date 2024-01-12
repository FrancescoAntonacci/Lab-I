import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

## Mother Fucker func

def onclick(event):
    print(round(event.xdata),",",round(event.ydata))

## Image
file_path = r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Catenary\catenary1.jpg"

fig=plt.figure("Immagine originale")
img = matplotlib.image.imread(file_path)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.imshow(img)
plt.show()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

## Catenary
def catenary(x, a, c, x0):
    '''Model of a catenary.
    '''
    return c + a * np.cosh((x - x0)/ a)

file_path = r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Catenary\catenary.txt"
##  Parabola

# def parabola(x, a, b, c):
#     '''Model of a parabola.
#     '''
#     return a*x*x+ b*x +c
##Data

x, y = np.loadtxt(file_path, unpack=True, delimiter=",")



dx=1.750
s_dx=0.005
dy=1.550
s_dy=0.005

kx=dx/(max(x)-min(x))
ky=dy/(max(y)-min(y))

x=x*(kx)
y=y*(ky)

sigma_x = 5*kx
sigma_y = 5*ky
##Best Fit-Algorythm for the Catenary

p0=(-480*ky, 2016*ky,1223*kx)
popt, pcov = curve_fit(catenary, x, y, p0)
a_hat, c_hat, x0_hat = popt
sigma_a, sigma_c, sigma_x0 = np.sqrt(pcov.diagonal())
print("a_hat=",a_hat,"m\n sigma_a=", sigma_a,"m\n c_hat=", c_hat,"m\n sigma_c=", sigma_c,"m\n x0_hat=", x0_hat,"m\n sigma_x0=", sigma_x0,"m")

##Best Fit-Algorythm for the Parabola
# popt_p, pcov_p = curve_fit(parabola, x, y,p0=(-1000,1000 ,2000))
# ap_hat, bp_hat, cp_hat = popt_p
# sigma_ap, sigma_bp, sigma_cp = np.sqrt(pcov_p.diagonal())
# print(ap_hat, sigma_ap, bp_hat, sigma_bp, cp_hat, sigma_cp)

##Residuals computation

res = y - catenary(x, a_hat, c_hat, x0_hat)

##Chi-Square test

chi_square=sum((res/sigma_y)**2)
print("Chi_square=",chi_square)

##Plots Catenary
plt.figure("Punti e grafico dei residui ")

plt.errorbar(x, y, sigma_y, fmt=".",label="Punti misurati")

x1=np.linspace(min(x),max(x),10000)

plt.plot(x1, catenary(x1, a_hat, c_hat, x0_hat),label="Best-Fit algorithm for a catenary")
plt.grid(True)

# plt.set(xlabel='$x [m]$', ylabel='$ y[m]$')
plt.legend()


##Residuals plots Catenary
plt.figure("Residui")



plt.grid(True)
plt.errorbar(x, res, sigma_y, fmt=".",label="Residuals")
# plt.set(xlabel='$x [m]$', ylabel='$ y[m]$')
plt.legend()


# ##Plots Parabola
# fig = plt.figure("Fit e residui per la parabola")
# fig.add_axes((0.1, 0.3, 0.8, 0.6))
# plt.errorbar(x, y, sigma_y, fmt=".")
#
#
# x1=np.linspace(min(x),max(x),10000)
#
# plt.plot(x1, parabola(x1, ap_hat, bp_hat, cp_hat))
# plt.grid(which="both", ls="dashed", color="gray")
# plt.ylabel("y [u. a.]")
# plt.xlabel("x [u. a.]")
# ##Residuals plots Parabola
# fig.add_axes((0.1, 0.1, 0.8, 0.2))
# res = y - parabola(x, ap_hat, bp_hat, cp_hat)
# plt.errorbar(x, res, sigma_y, fmt="o")
# plt.grid(which="both", ls="dashed", color="gray")
# plt.xlabel("x [u. a.]")
# plt.ylabel("Residuals")
# plt.ylim(-20.0, 20.0)
# plt.savefig("catenaria.pdf")






plt.show()
