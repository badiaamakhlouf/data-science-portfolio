# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:24:50 2017

@author: User
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import measure
import math



dataset = ["low_risk_1.jpg","low_risk_2.jpg","low_risk_3.jpg","low_risk_4.jpg","low_risk_5.jpg",
 "low_risk_6.jpg","low_risk_7.jpg","low_risk_8.jpg","low_risk_9.jpg","low_risk_10.jpg","low_risk_11.jpg","medium_risk_1.jpg",
 "medium_risk_2.jpg","medium_risk_3.jpg","medium_risk_4.jpg","medium_risk_5.jpg","medium_risk_6.jpg", 
 "medium_risk_7.jpg","medium_risk_8.jpg","medium_risk_9.jpg", "medium_risk_10.jpg","medium_risk_11.jpg","medium_risk_12.jpg","medium_risk_13.jpg"
  ,"medium_risk_14.jpg","medium_risk_15.jpg","medium_risk_16.jpg","melanoma_1.jpg","melanoma_2.jpg","melanoma_3.jpg",
  "melanoma_4.jpg","melanoma_5.jpg", "melanoma_6.jpg","melanoma_7.jpg","melanoma_8.jpg","melanoma_9.jpg",
  "melanoma_10.jpg", "melanoma_11.jpg","melanoma_12.jpg","melanoma_13.jpg","melanoma_14.jpg","melanoma_15.jpg", "melanoma_16.jpg",
  "melanoma_17.jpg","melanoma_18.jpg","melanoma_19.jpg","melanoma_20.jpg","melanoma_21.jpg", "melanoma_22.jpg" ,"melanoma_23.jpg",
  "melanoma_24.jpg","melanoma_25.jpg","melanoma_26.jpg","melanoma_27.jpg"]


ratio = np.zeros(len(dataset))
indexim = 0
for item in dataset:
         # Read original image 
        filein="C:\\Users\\User\\Dropbox\\3 ICT for Health\\3 ICT Health Labs\\ICT Health Lab4\\"+ item
        im = mpimg.imread(filein)
        
        plt.figure()
        plt.imshow(im)
        plt.title('original image')
        plt.show()
        red_amount= im[:,:,0] #is the amount of red color, from 0 to 255
        gree_amount= im[:,:,1] #is the amount of green color, from 0 to 255
        blue_amount= im[:,:,2] #is the amount of blue color, from 0 to 255
        #Value [0,0,0] corresponds to black, value [255,255,255]
        #corresponds to white
        
        # K-means on original image 
        number_clusters=3 # the number of clusters 
        kmeans = KMeans(n_clusters =number_clusters , random_state=0 )
        #kmeans.fit(im)  # after executing this part of script it arises an error because python requires 
        #a 2D Ndarray not a 3D Ndarray
        # reshape the array
        [N1, N2,N3]=im.shape
        im_2D=im.reshape((N1*N2, N3)) # N1*N2 rows and N3 columns
        [Nr, Nc]=im_2D.shape
        # K-means on 2D Ndarray
        kmeans.fit(im_2D)
        b=kmeans
        kmeans.cluster_centers_ # the centroids of the clusters
        labels=kmeans.labels_ #the  N1*N2 classes / clusters each pixel belongs to
        # the centroids are 3 NDarrays of float numbers, but we need
        #uint8 numbers to show the image; therefore the centroids become:
        centroids=kmeans.cluster_centers_.astype ('uint8')
        #These three centroids represent the three colors that k-means found as
        #representatives of all the image colors (the original image has
        #potentially 2^24 = 16 *10^6 different colors, but we want only 3 different colors)
        
        imm=im_2D.copy()
        imm_sci=imm.reshape((N1*N2, N3))
        #kmeans_centroids= kmeans.cluster_centers_. astype
        
        for kc in range (number_clusters):
             ind=(kmeans.labels_==kc)
             #imm_sci[ind,:]=centroids [kc,:]
             imm_sci[ind,:]=centroids[kc,:]
        # to show the image we need 3D Ndarray not 2D,  imm_sci is 3D must be transformed into 2D     
        im3D = imm_sci.reshape((N1,N2,N3))
        i=80
        im3D = im3D[i:N1-1-i,i:N2-1-i,:]
        [N1,N2,N3]=im3D.shape
        
        plt.figure()
        plt.imshow(im3D)#, interpolation=None)
        plt.title('the result of scikit kmeans')
        plt.show()
        sum=[0,0,0]
        j=0
        # 1) find the darkest color it corresponds to the mole :
        # I had used this Relative luminance standard which consists 
        #that the lowest value corresponds to the minimum sum  means this point has the lowest luminanace
        # so select the index because we are intersted to the darkest point 
        for j in range (3):
            sum[j]= 0.2126*centroids[j,0] + 0.7152*centroids[j,1] + 0.0722*centroids[j,2]
        
        darkIndex = np.argmin(sum)
        darkest_color=centroids[darkIndex,:]
        
        new_imag = np.zeros((N1,N2))
        for column in range(N2):
           for row in range(N1):
              if (im3D[row,column,:]- centroids[darkIndex,:]).all()==0:
                  new_imag[row,column] = 1
              
        plt.matshow(new_imag,cmap='Blues')
        plt.colorbar()
        plt.title('Mole Area')
        plt.axis('off')
        plt.show()
        # algorithm ipmlementation 
        # Opening removes small objects
        # Dilation: maximum filter:
        separated_im=new_imag.copy()
        step = ndimage.binary_erosion(separated_im)
        step1= ndimage.binary_opening(step, structure=np.ones((13,13))).astype(np.int)
        step2= ndimage.binary_dilation(step1).astype(separated_im.dtype)      
        # Separate the mole from the rest of the image
        plt.matshow(step2,cmap='Blues')
        plt.colorbar()
        plt.title('Filtered Mole Area ')
        plt.axis('off')
        plt.show()

        step3=step2.copy()
        x,y = np.nonzero(step3)
        step3=step2[np.min(x):np.max(x), np.min(y):np.max(y)]
        # find the median  and Calculate the center of the mole
        median_region=ndimage.measurements.center_of_mass(step3) 
        center=[i for i in median_region]
        center=[int(center[0]), int(center[1])]
        '''
        plt.matshow(step3,cmap='Blues')
        plt.plot(center[1], center[0], 'ro')
        plt.colorbar()
        plt.title('Filtered Mole with median ')
        plt.axis('off')
        plt.show()'''
        
        # the algorime
        # find the median  and Calculate the center of the mole
        '''median_region1=ndimage.measurements.center_of_mass(step2) 
        center2=[i for i in median_region1]
        center2=[int(center2[0]), int(center2[1])]'''
        
        index = []
        for j in range(0,N1):
             for k in range(0,N2):
                 if step2[j,k] == 1:
                    index.append((j,k))
        center2 = np.median(index, axis=0)
        center2=center2.astype(int)
        # Opening removes small objects
        # Separate the mole from the rest of the image
        plt.matshow(step2,cmap='Blues')
        plt.plot(center2[0], center2[1], 'ro')
        plt.colorbar()
        plt.title('Filtered Mole with median ')
        plt.axis('off')
        plt.show(block=False)      
        
        above=0
        left=0
        right=0
        leftdist=[] 
        abovedist=[] 
        rightdist=[]
        #while Repeat:
        # compute for above
        for above in range(center2[0]):
             if (step2[center2[0]-above, center2[1]]==1):
                 for left in range(center2[1]):
                    if (step2[center2[0]-above, center2[1]-left]==1):
                        leftdist.append(left)        
                 for right in range (len(step2[0])-center2[1]):
                     if (step2[center2[0]-above, center2[1]+right]==1):
                         rightdist.append(right)
                 
             abovedist.append(above)
             left=0
             right=0
        
        '''print(upperdist)
        print(rightdist)
        print(leftdist)'''
        
        under=0
        right1=0
        left1=0
        
        underdist=[]
        right1dist=[]
        left1dist=[]
             
        for under in range(len(step2)-center2[0]):
             if (step2[center2[0]+under, center2[1]]==1):
                 for left1 in range(center2[1]):
                    if (step2[center2[0]+under, center2[1]-left1]==1):
                         left1dist.append(left1)
                 for right1 in range(len(step2[1])-center2[1]):
                     if (step2[center2[0]+under, center2[1]+right1]==1):
                         right1dist.append(right1)
                 
                 underdist.append(under)
                 right1=0
                 left1=0
        '''print(underdist)
        print(rightdist2)
        print(leftdist2)'''
        maxright=right1dist+rightdist
        maxleft=left1dist+leftdist
        bothleft_right=maxright+maxleft
        bothabove_under= underdist +abovedist
        
        moverow=center2[0]-np.max(bothabove_under) #==> xxxxxx 
        movecol=center2[1]-np.max(maxleft)   #==> yyyyy
        
        ony= center2[1]+np.max(maxright)# column
        onx=center2[0]+np.max(bothabove_under) # row
        
        # I decide to leave a magre of 20
        step4=step2[int(moverow): int(onx), int(movecol): int(ony) ]
        
        newcenter=[center2[0]-moverow, center2[1]- movecol]
        
        #center2[0]is for x movements means for column movements
        #center2[1] is for x movement means for row movements
        # Opening removes small objects
        # Separate the mole from the rest of the image
        plt.matshow(step4,cmap='Blues')
        plt.plot(newcenter[1], newcenter[0], 'ro')
        plt.colorbar()
        plt.title('Filtered centred Mole with median ')
        plt.axis('on')
        plt.show(block=False)

        #Calculate the contour
        #method one
        # Find contours at a constant value of 0.8
        contours = measure.find_contours(step4, 0.8) 
        Num_countr_point=np.size(contours)
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(step4, interpolation='nearest', cmap='Blues')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        #method two
        [row,column] = step4.shape
        my_contour = np.zeros((row,column))
        for col in range(0,column):
           indexrow = 0
           condition_first=True # first pixel is 0
           
           while  condition_first :
               if step4[indexrow, col]==1:
                  my_contour[indexrow,col] = 1
                  condition_first=False
               else:
                   indexrow = indexrow+1
                   if indexrow == row:
                       condition_first=False
                       
        indexrow=0
        condition_second=True # last pixel is 1            
        for col in range(0,column):                
            while condition_second:
                if step4[row-indexrow-1,col] == 1:
                   my_contour[row-indexrow-1,col] = 1
                   condition_second=False
                else:
                    index_row = indexrow-1
                if indexrow == 0:
                           condition_second=False
        
        
        for ro in range(0,row):
           firstcond = True
           secondcondition = True
           colindex = 0
           while firstcond :
               if step4[ro,colindex] == 1:
                  my_contour[ro,colindex] = 1
                  firstcond = False    
               else:
                   colindex = colindex+1
               if colindex == column:
                    firstcond = False    
           colindex =column-1
           while secondcondition:
               if step4[ro,colindex] == 1:
                       my_contour[ro,colindex] = 1
                       secondcondition = False
               else:
                   colindex = colindex-1
               if colindex == 0:
                   secondcondition = False
                       
        plt.matshow(my_contour,cmap='Blues')
        plt.title('Contour of the mole')
        plt.colorbar()
        plt.axis('off')
        plt.show(block=False)
        
        
        
        # calculate the ratio 
        perimeter_mole= np.sum(np.sum(my_contour) )# perimeter of circle is the number of lines in the contour 
        # compute the number of point have value 1 from the image
        Area=0
        [new_row, new_col]=step4.shape
        # to compute the perimeter of mole I had used the area of the circle and I had performed this two for loop
        #to calculate the area which is equal to number of pixels assigned to 1
        for badia in range(0, new_row):
           for  badia2 in range (0,new_col):
                if step4[badia, badia2] ==1:
                    Area=Area+1
        
        perimeter_circle = 2*math.sqrt(Area*math.pi) # perimeter of mole which is a circle
        
        ratio[indexim] =perimeter_circle /perimeter_mole
        print(item)
        print(ratio[indexim])
        indexim = indexim + 1
        # to improve the contour detection 



