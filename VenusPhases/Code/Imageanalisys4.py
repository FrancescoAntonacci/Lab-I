##Magical libraries!

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

## Probably these are the only few lines in this code i can fully understand...

def distance_squared(x,y,x_c,y_c):
    d=(x-x_c)**2+(y-y_c)**2
    return d

def area(r,s_r):
    area=[]
    s_area=[]
    for i in range(0,np.size(r)):
        area.append(np.pi*r[i]**2)
        s_area.append(np.pi*2*s_r[i])
    return area, s_area

def ratio(a,b,s_a,s_b):
    r=[]
    s_r=[]
    for i in range(0,np.size(a)):
        r.append(a[i]/b[i])
        s_r.append((s_a[i]/a[i]+s_b[i]/b[i])*r[i])
    return r,s_r


## KASA FITTTT FOR CIRCLE BLYAT!!!!
def fit_circle(x, y, sigma):
    """Fit a series of data points to a circle.
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
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a binary mask based on the brightness threshold
    mask = (image > brightness_threshold).astype(np.uint8)

    # Count the number of pixels above the threshold
    above_threshold_count = np.sum(mask)

    return above_threshold_count


##

radii=[] #apparent radii of Venus in the photos
s_radii=[] #related uncertainty

phi=[]#Number of pixel above the threshold of luminosities in the photos
s_phi=[] #related uncertainty
##

manp=14 # which photo to end
minp=1 #which photo to star

##Loading images from file


##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!

folder_path=r'C:\Users\zoom3\Documents\Unipi\Laboratorio I\VenusPhases\venus'

##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!CHANGE THE DIRECTORY!!!!!!!!!!!!!!!!!

for i in range (minp, (manp+1),1):
    print("Photo:",i)
    num=str(i)

    image_path=folder_path+num+'.jpg'
    image = cv2.imread(image_path)
 ##Mother-fucking image reading function

    image_original=image

    image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE) #50 shades of Grey


    ## Apply MMMMMMMMMMMOOORE contrast for the meme
    min_intensity = 100 # Adjust as needed
    max_intensity = 400  # Adjust as needed
    contrast_stretched = np.clip((image - min_intensity) * (255.0 / (max_intensity - min_intensity)), 0, 255).astype(np.uint8)
    image=contrast_stretched

    figure_index1=2*i-1
    figure_index2=2*i


    ## Apply edge detection with our best friend Canny!!!

    edges = cv2.Canny(image, 1000, 0)
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0,255, 0), 2)

    # Extract and print coordinates of contours
    x=[]
    y=[]
    for i, contour in enumerate(contours):
        # print(contour)
        coordinates = contour.reshape(int(contour.size/2),2)
    for j in range(0,int(np.size(coordinates)/2)-1):
        x.append(coordinates[j,0])
        y.append(coordinates[j,1])


##Point/politicians killing algorithm
    x_c=[]
    y_c=[]
    x_e=[]
    y_e=[]
    for i in range(min(y),max(y)):
        giudici=[]
        for j in range(0,int(np.size(y))):
            if(y[j]==i):
                giudici.append(x[j])
        # print(i,giudici)
        if (np.size(giudici)>0):
            x_c.append(max(giudici))
            x_e.append(min(giudici))
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

    ## not necessary plots
    # plt.figure(figure_index1)
    # plt.imshow(image_original)
    # plt.errorbar(x_c,y_c,fmt='.')
    # plt.errorbar(x_e,y_e,fmt='.')
    # plt.figure(figure_index2)
    # plt.imshow(contour_image)
    # plt.show()# # Parametri dell'ellisse
##

    fig, ax = plt.subplots()

    PALLA = patches.Circle([x_circle,y_circle],radius=r,alpha=0.5)
    ax.plot(x_c,y_c)
    ax.add_patch(PALLA)
    ax.imshow(image_original)
    ax.errorbar(x_c,y_c,fmt='.')
    plt.show()

    ## Get that radius!
    radii.append(r)
    s_radii.append(sigma_r)

## Luminosity test


for i in range (minp, (manp+1),1):
    print("Photo:",i)
    num=str(i)

    image_path=folder_path+num+'.jpg'
    image = cv2.imread(image_path)

    image_original=image

    # Specify the image path and brightness threshold
    brightness_threshold = 50  # Adjust this value as needed

    # Count pixels above the threshold using OpenCV
    count = count_bright_pixels_cv(image_path, brightness_threshold)

    phi.append(count)
    s_phi.append(np.sqrt(count)) # to be adjusted since it is completely arbitrary
    print(f"Number of pixels above brightness threshold: {count}")



##

app_area_venus,sigma_apparent_area=area(radii,s_radii)

ratiod_i,s_ratio_i=ratio(phi,app_area_venus,s_phi,sigma_apparent_area)
perc_i=np.multiply(ratiod_i,100)
s_perc_i=np.multiply(s_ratio_i,100)

## WONNDERFULL PLOTTINO

plt.figure("Plottino bello")
plt.errorbar(perc_i,app_area_venus,sigma_apparent_area,s_perc_i, fmt='o',label="Punti misurati")
plt.grid(True)
plt.xlabel('$Fase \% $')
plt.ylabel('$Superficie apparente di Venere[pixel^2]$')
plt.legend()
plt.show()






## Plots to get to understand something (even though they are quite useless by themselves)

# che_palle=np.linspace(minp,manp,(manp-minp+1))
#
# plt.figure()
# plt.plot(che_palle,phi)
# plt.show()


## ...


# plt.figure()
# plt.plot(che_palle,app_area_venus)
# plt.show()
