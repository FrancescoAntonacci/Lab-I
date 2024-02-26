##Magical libraries!

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit


## DECLARATION OF INTENTIONS

lum_plots=False# turn on luminosity distribution plots writing "True"
cir_plots=False # turn on writing "True"

## Probably these are the only few lines in this code i can fully understand...

def distance_squared(x,y,x_c,y_c):
    '''
    Don't be silly.
    '''

    d=(x-x_c)**2+(y-y_c)**2
    return d

def area(r,s_r):
    '''
    Don't be silly.
    '''
    area=[]
    s_area=[]
    for i in range(0,np.size(r)):
        area.append(np.pi*r[i]**2)
        s_area.append(np.pi*2*s_r[i])
    return area, s_area

def stupid_area(r):
    '''
    same as before but idiotic
    '''
    a=np.pi*r**2
    return a

def ratio(a,b,s_a,s_b):
    '''
    Don't be silly.
    '''
    r=[]
    s_r=[]
    for i in range(0,np.size(a)):
        r.append(a[i]/b[i])
        s_r.append((s_a[i]/a[i]+s_b[i]/b[i])*r[i])
    return r,s_r


##Fit law
def odiovenere(phi,st,vs,r0):
    A=(np.pi*(r0)**2)/(vs**2+st**2-2*(1-2*phi)*vs*st)
    return A


## Missing part:
#The code will be completed once it will incude the law and the x^2 test.
#That is the simplest part of it all!
##


## KASA FITTTT FOR CIRCLE BLYAT!!!!
def fit_circle(x, y, sigma):
    """Fit a series of data points to a circle. I have no idea of how this works, but I'm content saying that this works.
    """
    n = len(x)
    # Refer coordinates to the mean values of x and y.
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    # Calculate all the necessary sums.
    s_u = np.sum(u)
    s_uu = np.sum(u**2.0)
    s_uuu = np.sum(u**3.0)
    s_v = np.sum(v)
    s_vv = np.sum(v**2.0)
    s_vvv = np.sum(v**3.0)
    s_uv = np.sum(u * v)
    s_uuv = np.sum(u * u * v)
    s_uvv = np.sum(u * v * v)
    D = 2.0 * (s_uu * s_vv - s_uv**2.0)
    # Calculate the best-fit values.
    u_c = (s_vv * (s_uuu + s_uvv) - s_uv * (s_vvv + s_uuv)) / D
    v_c = (s_uu * (s_vvv + s_uuv) - s_uv * (s_uuu + s_uvv)) / D
    x_c = u_c + x_m
    y_c = v_c + y_m
    r = np.sqrt(u_c**2.0 + v_c**2.0 + (s_uu + s_vv) / n)
    # Calculate the errors---mind this is only rigorously valid
    # if the data points are equi-spaced on the circumference.
    sigma_xy = sigma * np.sqrt(2.0 / n)
    sigma_r = sigma * np.sqrt(1.0 / n)
    return  x_c, y_c, r, sigma_xy, sigma_r

## Brightness counting!

def count_bright_pixels_cv(image_path, brightness_threshold):
    '''
    Super boring function, give her the path and the threshold: she well return you the number of pixel she counted above the threshold.
    '''


    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a binary mask based on the brightness threshold
    mask = (image > brightness_threshold).astype(np.uint8)

    # Count the number of pixels above the threshold
    above_threshold_count = np.sum(mask)

    return above_threshold_count


def lum_distribution(image_path, brightness_threshold):
    '''
    Suggested by our lord T-man


    '''

    lum_dis=[]

    for i in range(0, brightness_threshold+1):

        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Create a binary mask based on the brightness threshold
        mask = (image > i).astype(np.uint8)

        # Count the number of pixels above the threshold
        atc = np.sum(mask) #above threshold count (ascending order)

        lum_dis.append(atc)



    return lum_dis/max(lum_dis)

## :-)

def mother_fucking_func(minp,manp,folder_path,plots):
    '''
    Mother_fucking_func does what it sounds like.
    First it will find you, then it will ask you the minimum and maximum index of the photos to analyze, please give him the folder and for courtesy tell him if you whant to have some plots. It will return, after many traumas, the array of the exteemed radii and uncertainty, the number of pixels brighter than a threshold and uncertainty of Venus from your photos. Please be kind with this monster.
    '''



    for i in range (minp, (manp+1),1):
        print("Photo:",i)
        num=str(i)


        image_path=folder_path+num+'.jpg'
        image = cv2.imread(image_path)
    ##Mother-fucking image reading function

        image_original=image

        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE) #50 shades of Grey


        ## Apply MMMMMMMMMMMOOORE contrast for the meme  -- Generally it is a bad idea
        # min_intensity = 100 # Adjust as needed
        # max_intensity = 300  # Adjust as needed
        # contrast_stretched = np.clip((image - min_intensity) * (255.0 / (max_intensity - min_intensity)), 0, 255).astype(np.uint8)
        # image=contrast_stretched

        figure_index1=2*i-1
        figure_index2=2*i


        ## Apply edge detection with our best friend Canny!!!

        ##      10, 90 is quite working...
        ##
        edges = cv2.Canny(image, 10, 80)
        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0,255, 0), 2)

        # Extract and print coordinates of contours
        x=[]
        y=[]

        for contour in contours:
            for point in contour:
                xi, yi = point[0]
                x.append(xi)
                y.append(yi)

    ##Point/politicians killing algorithm
        x_c=[]
        y_c=[]
        x_e=[]
        y_e=[]
        for i in range(min(y),max(y)):
            giu=[]
            for j in range(0,int(np.size(y))):
                if(y[j]==i):
                    giu.append(x[j])
            # print(i,giu)
            if (np.size(giu)>0):
                x_c.append(max(giu))
                x_e.append(min(giu))
                y_c.append(i)
                y_e.append(i)

    ## The killig of those points that previously didn't cease to exist
        x_circle, y_circle, r, sigma_xy, sigma_r=fit_circle(x_c,y_c,4) #FIT!

        for maremma_cignala_voglio_i_dati_boni in range(0,5):

            mannboy=np.linspace(np.size(x_c)-1,0,np.size(x_c))
            for amate_il_toma_toma in mannboy: #amate_il_toma_toma, eternity, more or less.

                mask=distance_squared(x_c[int(amate_il_toma_toma)],y_c[int(amate_il_toma_toma)],x_circle,y_circle)<(0.99*r**2)
                if (mask):
                    x_c.pop(int(amate_il_toma_toma)) # Via il lezzume!
                    y_c.pop(int(amate_il_toma_toma))
            x_circle, y_circle, r, sigma_xy, sigma_r=fit_circle(x_c,y_c,1) #Fit, again!

    ## Happy plottin'
        if(plots==True):
            fig, ax = plt.subplots()

            PALLA = patches.Circle([x_circle,y_circle],radius=r,alpha=0.5)
            ax.plot(x_c,y_c)
            ax.add_patch(PALLA)
            ax.imshow(image)
            ax.errorbar(x_c,y_c,fmt='.')
            plt.show()

        ## Get that radius!
        radii.append(r)
        s_radii.append(sigma_r)
        print("Radius=",round(r,2),"$\pm$",round(sigma_r,2))
    ## THIS IS ALL YOU GET
    ##
    return radii,s_radii
    ##
    ##


##
def lumtest(plots):

    # Specify the image path and brightness threshold

    bc=20
    brightness_threshold = bc  # Adjust this value as needed!!



    lum_dis_all=[]#Luminosity distribution of all photos

    for i in range (minp, (manp+1),1):
        print("Photo:",i)
        num=str(i)


        image_path=folder_path+num+'.jpg'
        image = cv2.imread(image_path)

        image_original=image

        if (i<9):
            brightness_threshold=17 ##ADJUST FOR LESS WHEN THERE IS A LACK OF LUMINOSITY
        # Count pixels above the threshold using OpenCV
        if (i<3):
            brightness_threshold=14 ##ADJUST FOR LESS WHEN THERE IS A LACK OF LUMINOSITY
        # Count pixels above the threshold using OpenCV
        count = count_bright_pixels_cv(image_path, brightness_threshold)

        phi.append(count)
        s_phi.append(np.sqrt(count)) # to be adjusted since it is completely arbitrary

        print(f"Number of pixels above brightness threshold: {count}")
        brightness_threshold = bc  # Adjust this value as needed!!


        lum_dis=lum_distribution(image_path, brightness_threshold)
        lum_dis_all.append(lum_dis)
        if (plots==True):
            ## "thy shall plot the luminosity distribution" as our lord Toma Toma said

            x_plot=np.linspace(0,brightness_threshold,brightness_threshold+1)
            ##Happy plottin'
            name_plot="Luminoisty distribution of photo:"+num


            plt.figure(name_plot)
            plt.plot(x_plot,lum_dis)
            plt.show()
            ##

    x_plot=np.linspace(0,brightness_threshold,brightness_threshold+1)
    ## general plot of luminosity distribution



    plt.figure("General plot of luminosity distribution")
    print(np.size(lum_dis_all[0]))

    #
    # print(x_plot)
    # print(lum_dis_all)
    # print(np.size(lum_dis_all[0,0]))
    j=1
    for i in lum_dis_all:
        iteration=str(j)
        l_p="Photo:"+iteration # label of the plot
        plt.plot(x_plot,i,label=l_p)
        plt.xlabel("Luminosity Threshold")
        plt.ylabel("Percentage above threshold of the entire photo")
        j=j+1
        plt.legend()

    plt.show()

    return phi,s_phi
##




## Important variables

radii=[] #apparent radii of Venus in the photos
s_radii=[] #related uncertainty

phi=[]#Number of pixel above the threshold of luminosities in the photos
s_phi=[] #related uncertainty
##

manp=14 # which photo to end
minp=1 #which photo to start



##Loading images from file


##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!

folder_path=r'C:\Users\zoom3\Documents\Unipi\Laboratorio I\VenusPhases\code\venus'

##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!





## THE GREAT CALLc
radii,s_radii=mother_fucking_func(minp,manp,folder_path,cir_plots) ## TURN OFF PLOTS CHANGING LAST VARIABLE TO FALSE!


## the little call

lumtest(lum_plots) ## TURN OFF PLOTS CHANGING LAST VARIABLE TO FALSE!



## FINALLY, THE WONNDERFULL PLOTTINO IS COMING!

app_area_venus,sigma_apparent_area=area(radii,s_radii)

## let's put areas in a ratio with the maximum value
raav=app_area_venus/max(app_area_venus)
sraav=sigma_apparent_area/max(app_area_venus)


ratiod_i,s_ratio_i=ratio(phi,app_area_venus,s_phi,sigma_apparent_area)




##Correlation phase-apparent area,
cpa=np.corrcoef(ratiod_i,raav)
print("Correlation phase-apparent area",round(cpa[1,0],2))
## Finally we can FIIIT


popt, pcov = curve_fit(odiovenere, phi, raav,p0=(1,0.7,0.35), absolute_sigma=True)

st,vs,r0= popt

s_st, s_vs,s_r0= np.sqrt(pcov.diagonal())

print("st=",st,"+-",s_st,"vs=",vs,"+-",s_vs,"r0=",r0,"+-",s_r0)

# ##Residuals computation
#
# res = area_app -
# #
# # #Chi-Square test
# #
# # chi_square=sum((res/sigma_apparent_area)**2)
# # print("Chi_square=",chi_square)
# #

## some random data - to be deleted

phi=np.linspace(0.1,1)
d_model=odiovenere(phi,st,vs,r0)
# d_model=odiovenere(phi,1,0.76,0.35)

## WONNNDERFUL PLOTTINO
plt.figure("Plottino bello")


plt.errorbar(ratiod_i,raav,sraav,s_ratio_i, fmt='o',label="Punti misurati")

plt.errorbar(phi,d_model,label="Modello predittivo")

plt.grid(True)
plt.xlabel('$\Phi$')
plt.ylabel('$Superficie apparente di Venere (rapporto con la massima)$')
plt.legend()



plt.show()


