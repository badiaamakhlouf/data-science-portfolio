# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:33:31 2017

@author: User
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# defining a function to generate
# the same random vector at each time I run the script 
def Same_Random():
    random.seed(9001)
    rd = [random.randint(1,100) for i in range(1,22)]
    a = np.array( rd )
    return a

xx=pd.read_csv('parkinsons_updrs.data')
xx.info()
xx.describe().T
xx.test_time=xx.test_time-xx.test_time.min()
xx.test_time=xx.test_time.round()
xx.describe().T
Npatients=xx.loc[:,'subject#'].max()
data=xx.groupby(['subject#','test_time'],as_index=False).mean()
              
# considering only the first 36 patient
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
y_train=data_train_norm_values[:,5]
y_test=data_test_norm_values[:,5]
#deleting the %jitter column 
X_train=np.delete(data_train_norm_values,5,1)
X_test=np.delete(data_test_norm_values,5,1)


#%%
# using the Steepest Descent algorithm 
w_hat_old = np.ones(len(data_train_norm.columns)-1)
w_hat =Same_Random() # generating the same random value
sss=w_hat
y_hat_train= np.dot(X_train,w_hat_old)
#train_error = np.linalg.norm(np.dot(X_train,w_hat)-y_train)**2
# calculating the gradient 
X_train_transpose = X_train.T
#hyp=np.dot(X_train, w_hat)
#v = -y_train+hyp
gradient_w_hat = -2*np.dot(X_train.T,y_train ) + 2*np.dot(X_train.T,np.dot(X_train, w_hat))
# hessian matrix at point w_hat 
Hessian= 4*np.dot( X_train_transpose,X_train )
iterations=0
max_iterations = 1e4
old_error=[]
while np.linalg.norm(w_hat- w_hat_old) > 1e-8 and iterations < max_iterations:
      iterations += 1
      #old_error += [train_error]                                           
      w_hat_old=w_hat
      gamma=np.linalg.norm(gradient_w_hat)**2/np.dot(np.dot(gradient_w_hat.T,Hessian),gradient_w_hat)
      w_hat = w_hat - gamma * gradient_w_hat # update the guess 
      #train_error=np.linalg.norm(np.dot(X_train,w_hat)- y_train)**2
      #hyp=np.dot(X_train, w_hat)
      #v = -y_train+hyp
      gradient_w_hat = -2*np.dot(X_train.T,y_train ) + 2*np.dot(X_train.T,np.dot(X_train, w_hat))

y_hat_train= np.dot(X_train,w_hat)

y_hat_test = np.dot(X_test,w_hat)

# computing the square error for train
square_error_train_SD=np.power((y_hat_train-y_train),2)

#computing the square error for test 
square_error_test_SD=np.power((y_hat_test-y_test),2)   
   
# figure 1 : y_hat_train versus y_train
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(y_hat_train,y_train,'b*', label='y_hat_train_Steepest Descent Algo')
plt.plot(y_train,y_train,'r-', label='y_train')
plt.title('y_hat_train versus y_train')
plt.xlabel('y_hat_train/y_train')
plt.ylabel('y_train')
plt.legend(loc=2)

# figure 2: y_hat_test versus y_test
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(y_hat_test,y_test,'ro', label='y_hat_test_Steepest Descent')
plt.plot(y_test,y_test,'b-', label='y_test')
plt.xlabel('y_hat_test / y_test')
plt.ylabel('y_test')
plt.title('y_hat_test versus y_test -Steepest Descent Algo')
plt.legend(loc=2)

g1=y_train-y_hat_train
#figure 3 : histograms of y_train-y_hat_train
#plt.figure(figsize=(13,6)) 
plt.figure()
plt.hist(g1, bins=50,range=[-0.2,0.5])
plt.title("Train - Error histogram-Steepest Descent Algo")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)

#figure 4 : histograms of y_test-y_hat_test
#plt.figure(figsize=(13,6))
plt.figure()
plt.hist(y_test-y_hat_test,bins=50, range=[-0.2,0.5])
#xticks(range(0,0.5))
plt.title("Test - Error histogram-Steepest Descent Algo")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)            


#figure 5 : plotting the values of w_hat
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(w_hat,'b*')
plt.title("Regression coefficients-Steepest Descent Algo")
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.ylim((-27,27))
plt.legend(loc=2)
plt.grid


#figure 6 : plotting the values of square error for train set
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(square_error_train_SD,'b', label='Steepest descent')
plt.title("square error for the Train data")
plt.xlabel("rows number of train data")
plt.ylabel("Train Error")
plt.ylim((0,0.8))
plt.legend(loc=2)
plt.grid
plt.show()

#figure 7 : plotting the values of square error for test
#plt.figure(figsize=(13,6))
plt.figure()
plt.plot(square_error_test_SD,'b', label='Steepest Descent')
plt.title("square error for the test data")
plt.xlabel("rows number of  the Test data-")
plt.ylabel("Test-Error")
plt.ylim((0,0.75))
plt.legend(loc=2)
plt.grid
plt.show()
#%%