# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:33:31 2017

@author: Badiaa
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

xx=pd.read_csv('parkinsons_updrs.data')
xx.info()
xx.describe().T
xx.test_time=xx.test_time-xx.test_time.min()
xx.test_time=xx.test_time.round()
xx.describe().T
Npatients=xx.loc[:,'subject#'].max()
data=xx.groupby(['subject#','test_time'],as_index=False).mean()
              
# onsidering only the first 36 patient
# creating the data train 
index_train= data['subject#']<=36
data_train=data[index_train]
# creating the data test 
index_test=data['subject#']>36
data_test=data[index_test]
# normalizing the data 
# m and s are those of the training matrix
m=data_train.mean() # mean (one valye for each column)
s=data_train.std() #standard deviation (one value for each column)
data_train_norm=(data_train-m)/s # normalized training data 
data_test_norm=(data_test-m)/s #normalized testing data 


#changing the Pandas data frame to Numpy arrays
data_train_norm_values=data_train_norm.values
data_test_norm_values=data_test_norm.values          
  # regression 
  # the feature that we want to estimate is Y_train and y_test
y_train=data_train_norm_values[:,5] # 5 corresponds to the total_UPDRS and 6 corresponds to the jitter
y_test=data_test_norm_values[:,5] # 5 corresponds to the total_UPDRS and 6 corresponds to the jitter
#deleting the %jitter column 
X_train=np.delete(data_train_norm_values,5,1)
X_test=np.delete(data_test_norm_values,5,1)

#MSE Regression    
# train data     
W_hat= np.dot(np.linalg.pinv(X_train), y_train)  

y_hat_train= np.dot(X_train,W_hat)

y_hat_test = np.dot(X_test,W_hat)
# computing the square error for train
square_error_train_MSE=np.power((y_hat_train-y_train),2)

#computing the square error for test 
square_error_test_MSE=np.power((y_hat_test-y_test),2)

# figure 1 : y_hat_train versus y_train
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(y_hat_train,y_train,'b*', label='y_hat_train_MSE')
plt.plot(y_train,y_train,'r-', label='y_train')
plt.title('y_hat_train versus y_train')
plt.xlabel('y_hat_train/y_train')
plt.ylabel('y_train')
plt.legend(loc=2)

# figure 2: y_hat_test versus y_test
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(y_hat_test,y_test,'ro', label='y_hat_test_MSE')
plt.plot(y_test,y_test,'b-', label='y_test')
plt.xlabel('y_hat_test / y_test')
plt.ylabel('y_test')
plt.title('y_hat_test versus y_test -MSE')
plt.legend(loc=2)

g1=y_train-y_hat_train
#figure 3 : histograms of y_train-y_hat_train
#plt.figure(figsize=(13,6))
plt.figure() 
plt.hist(g1, bins=50,range=[-0.2,0.5])
plt.title("Train - Error histogram-MSE")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)

#figure 4 : histograms of y_test-y_hat_test
#plt.figure(figsize=(13,6))
plt.figure()
plt.hist(y_test-y_hat_test,bins=50, range=[-0.2,0.5])
#xticks(range(0,0.5))
plt.title("Test - Error histogram-MSE")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)            


#figure 5 : plotting the values of w_hat
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(W_hat,'b*')
plt.title("Regression coefficients-MSE")
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.ylim((-27,27))
plt.legend(loc=2)
plt.grid

#figure 6 : plotting the values of square error for train set
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(square_error_train_MSE,'b', label='MSE')
plt.title("square error for the Train data")
plt.xlabel("rows number of train data")
plt.ylabel("Train Error")
plt.ylim((-0.01,0.9))
plt.legend(loc=2)
plt.grid
plt.show()

#figure 7 : plotting the values of square error for test
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(square_error_test_MSE,'b', label='MSE')
plt.title("square error for the test data")
plt.xlabel("rows number of  the Test data-")
plt.ylabel("Test-Error")
plt.ylim((-0.01,0.7))
plt.legend(loc=2)
plt.grid
plt.show()