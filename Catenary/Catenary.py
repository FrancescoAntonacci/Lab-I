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
xx=np.linspace(0,1800,100000)
yy=np.full_like(xx,2040)
plt.plot(xx,yy)
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

file_path = r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Catenary\catenary2.txt"
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

kx=dx/(2359-24)
ky=dy/(2123-9)

x=x*(kx)
y=y*(ky)

sigma_x = 1.15*kx##1.15= 2pixels/sqrt(12) ; where the two pixels is the uncertanty taking the data
sigma_y = 1.15*ky
##Best Fit-Algorythm for the Catenary

p0=(-10*ky, 10*ky,400*kx)
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
print("Chi_square=",chi_square,"\nDegrees of freedom=108")

##Plots Catenary
plt.figure("Punti e grafico dei residui ")

plt.errorbar(x, y, sigma_y, fmt=".",label="Punti misurati")

x1=np.linspace(min(x),max(x),10000)

plt.plot(x1, catenary(x1, a_hat, c_hat, x0_hat),label="Previsione dell'algoritmo di Best-Fit ")
plt.grid(True)
plt.xlabel('$x [m]$')
plt.ylabel('$ y[m]$')
plt.legend()


##Residuals plots Catenary
plt.figure("Residui")



plt.grid(True)
plt.errorbar(x, res, sigma_y, fmt=".",label="Residui")
plt.xlabel('$x [m]$')
plt.ylabel('$ y[m]$')
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
