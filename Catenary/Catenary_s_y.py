import numpy as np

path1=r"C:\Users\zoom3\Documents\Unipi\Laboratorio I\LaboratoryReports\Catenary\catenary_sy.txt"
x,y=np.loadtxt(path1 ,unpack=True, delimiter=",")
v=[]
for i in range (0,x.size-1):
    if(x[i]==1247):
        v.append(y[i])
v=np.array(v)
n=v.size
m=sum(v)/n
a=np.full_like(v,m)
a=v-a
s=sum(a**2)/(n-1)