# Melanoma cell detection using K-means and agglomerative clustering algorithms 
## Objectives: 
This work is composed by two parts: 

-The first part aims to help medical doctors in moles analysis by analyzing the border of each mole presented in a given image using the K-means clustering algorithm of Scikit learn. 

-The second part consists on performing the agglomerative clustering on the data related to chronic kidney disease and then results were commented. 

Knowing that, doctors to diagnose melanoma moles consider not only borders but also four other features: asymmetry, color, diameter and evolution. 

After performing the K-means algorithm on each of the given image (in data/images) and extracting the mole area, it was required to find the corresponding contour. Also, evaluate the perimeter of a circle that has an area equal to the mole area then evaluate the ratio between the perimeter of the mole and the perimeter of the circle. 
# Part 1: 
## 1.Dataset description: 
The provided dataset for this lab comports 54 images (in data/images) classified into three classes:
### Class 1: 11 images have the name low_risk_n.jpg for moles and they have a low probability of being melanoma (i.e. tumors).
### Class 2:  16 images named medium_risk_n.jpg for moles that have a low probability of be-ing melanoma.
### Class 3: 27 images of name_melanoma_n.jpg for moles that have a high probability of being melanoma.
In all the mentioned cases n was an integer. 
## 2.The K-means algorithm:
It is one of the iterative clustering algorithms, which consists on splitting the original da-taset into K clusters. Samples are assigned to each cluster based on the minimum distance between each sample and the cluster’s centroid. The centroid is updated for each iteration till a stop condition is verified. In this work, the cluster number was chosen 3 and the observations or samples were the pixels of the image. 

For each iteration, the K-means tests hundreds of different initial vectors to select the best clustering among the obtained results. The best clustering is when the initial vector has pre-sented the minimum moment of inertia. The performance of K-means or the best solution strictly depends on the target features. 

## 3.Requested image processing steps:
1- Read the image in Python and show it ==> the image is made of 583x583 pixels and each color has its value as a three element vector [a,b,c], e.g. black is [0,0,0] and white is [255,255,255]. Result is illustrated in Figure 1 : original image 

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2012%20original.png) .

2- Perform the K-means algorithm of Scikit learn library on the original image (Figure 1) setting the number of clusters to 3 as mentioned before but need to transform the image to 2D Ndarray. In order to show the image again reshape it to 3D Ndarray representation is in Figure 2 

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2013%20The%20result%20of%20scikit%20Kmeans.png)

3- Find the set of points forming the mole. Knowing that, those points have the darkest color. A relative luminance RGB (Read, Green, Blue) standard has been used which consists that the lowest value (0 value) corresponds to the darkest point. So, select its index. The result of selection is presented in the Figure 3 below.

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2014%20Mole%20Area%20after%20darkest%20points%20selection.png)

The centroids is a vector of 3 elements equal to the number of clusters, so the index j goes from 0 to 2
relative_luminance[j]= 0.2126*centroids[j,0] + 0.7152*centroids[j,1] + 0.0722*centroids[j,2]

4- In order to separate the mole from the rest of the image, the following steps were realized on the output of K-means:

	Erosion ==> removes small objects
	
	Opening ==> removes small objects for the second time with structure=np.ones((int,int))
	
	Dilation ==> maximum filter to make the boundaries smoother 
  
Results of Erosion, Opening and dilation:

From figures: 4, 5 and 6 it is clear that each time the filter (int(n,n)) increases the mole area is reduced. In the coming work, the int has been chosen 13 in order to be good and convenient for all other pictures.

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2015%20Filtered%20Mole%20Area%20with%20np.ones((20%2C20)).png)

Figure 4: Filtered Mole Area with np.ones((20,20))

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2016%20Filtered%20Mole%20Area%20with%20np.ones((13%2C13)).png)

Figure 5: Filtered Mole Area with np.ones((13,13))

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2017%20Filtered%20Mole%20Area%20with%20np.ones((4%2C4)).png)

Figure 6: Filtered Mole Area with np.ones((4,4))

Calculate the median, which corresponds to the center of the mole: consider all the pixels with value 1 then perform median of the founded 2D array using numpy the illustration is in Figure 7.

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2018%20Filtered%20mole%20with%20median.png)

Implement an algorithm that starts from the center of the mole, which was calculated through the median, and finds rectangular region that includes the entire mole to isolate the mole from the rest of the image. Figure 7.1 : Filtered centered Mole with median

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2019%20Filtered%20centered%20Mole%20with%20median.png)

5- Contour detection: considering each column and find the index of the first and last pixel with value 1 then perform the same work considering each row. 

The Figure 8 was obtained using “measure.find_contours “the predefined function of skimage library while the Figure 9 was obtained after performing the previous algorithm.

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2020%20Contour%20with%20%E2%80%9Cmeasure.find_contours%E2%80%9D%20function%20of%20skimage.png)

Figure 8: Contour with “measure.find_contours” function of skimage 

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/Figure%2021%20%20Contour%20of%20the%20mole.png)

Figure 9: Contour of the mole after applying my algorithm

6- Calculate the ratio between the mole perimeter and the contour perimeter 
The circle perimeter = √(4*π*Area), Area is the sum of all pixels with value 1 ( forming the mole)
The mole perimeter is the sum of all points in the contour
Ratio=  (perimeter_circle)/(perimeter_mole)

# Part 2: Agglomerative/ hierarchical clustering and chronic kidney disease 
## 1.The Chronic kidney disease (CKD) (From my professor's course):
A disease that affects the Kidney’s functionality. The kidney may loses its function for a period of months or years. This kind of disease has many causes where the major one is diabetes, high blood pressure, glomerulonephritis and polycystic kidney disease. Diagnosis is generally by blood tests to measure the glomerular filtration rate and urine tests to measure albumin.
## 2.Dataset description:
The first 29 rows of the provided dataset file for this work contain the description of the features and they must be skipped. The total dataset contains 400 instances classified into two classes as follow: 250 corresponding to the CKD and 150 not CKD.

From the provided description, it is noticed the existence of 24 features plus the last field (column 25) that corresponds to the class which specifies if the disease is present (ckd) or not (notckd). In the 24 features, it exists 11 numerical features and the rest are categorical or nominal features. 

Before the analyzing phase, the dataset must be prepared and cleaned due to the existence of some errors and missing fields. This task is usually needed and sometimes requires a long period. Moreover, the dataset contains some rows with an extra separator field “,”, so 26 columns were read instead of 25. The missing fields were identified by“?” and at the end the categorical features must be transformed into numerical but before that there are “hidden” typing error corresponds to typing “ yes” and not “yes” must be deleted.
Two options are exist for cleaning the data: manually by editing the original CSV file but it is preferable if the file is short while in our case it is better to exploit arguments of pandas.

Besides, the Scikit Learn implementation requires numerical data only so all categorical features must be mapped into numbers.

•	Yes ==> 1 and no ==> 0, Normal ==> 1 and abnormal==> 0, etc.

In order to manage the NaN values the following two approaches were applied:

•	Removing the rows containing NaN values using the methods dropna of Pandas (case 1). 

•	Treating NaN values as another possible random variable but must be substituted with a number not already presented in the dataset. In this report -3 was chosen as random value (case 2).

## 3. Agglomerative clustering
It starts from N observations (rows number of matrix X), each ob-servation is a vector of F features, I evaluated the distance between each two observations from the N observations. The two observations characterized by the minimum distance form the first cluster. Then I repeated the process with N-1 objects, which are N-2 observations and 1 cluster that is composed by 2 observations with first minimum distance. A repetitive process that stop when we get just one object.

In this second script, I had used the agglomerative clustering of SciPy and we had obtained the following result in Figure 10 and Figure 11. The result refers to the cases mentioned before remove and substitution.

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/case1.png) 

Figure 10: Removing the rows containing NaN values using the methods dropna of Pandas (case 1) 

![alt text](https://github.com/BaddyMAK/Clustering-with-ML/blob/main/results/case2.png)

Figure 11: Substitution of NaN by -3  










