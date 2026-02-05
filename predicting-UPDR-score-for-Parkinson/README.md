# Linear regression of the UPDR score for the Parkinson’s patients using: 
## MSE, Gradient Descent (GD) and Steepest Descent (SD)
## 1- Parkinson disease: 
Parkinson's disease affects the muscles of patients so they become unable to control them and they start suffering difficulties during walking or starting any kind of movement. Major part of those patients cannot speak correctly because they cannot control the vocal chords. 

In order to control the illness and to measure its severity, neurologists ask patients to perform many movements to collect scores together and to extract a final grade from them, which is, called UPDRS (Unified Parkinson's Disease Rating Scale).
## 2- Objectives and dataset description: 
The provided dataset contains biomedical voices for 42 patients diagnosed with Parkinson’s disease but in early stage. This set has been collected during almost six months with a rate once a week and composed by the following elements : Subject number, age, sex, test_time, motor-UPDRS, total_UPDRS and 16 biomedical voice measures: Jitter(%), Jitter: RAP, Jitter: PPQ5, Jitter:DDP, Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer: APQ5, Shimmer: APQ11, Shimmer:DDA, NHR, HNR, RPDE, DFA, PPE. Each parameter was characterized by 5-6 different measures per day while UPDRS and motor-UPDRS were measured only once a day. 

This work aims first to predict Jitter (%) then UPDRS using the remained features of biomedical voices. The prediction was done by means of the linear regression algorithms: MSE, GD and SD. To study the performance against over-fitting phenomenon the K-fold cross validation technique was applied. 
Moreover, this set of data must be prepared before using it: 

- Prepare a new matrix where test time (column 3) goes from 0 to 180 (only six months are considered) for each patient  after this step we get a matrix of 990 rows instead of matrix with 5786 rows (in original data set).

- Consider only data related to the first 36 patients as training set (store them in the matrix data_train) and leave patients from 37 to 42 for the testing phase (store them in the data_test).

- Normalize the two matrices: data_train and data_test and call them data_train_norm and 
data_test_norm: 

-	For the data_train: after normalization, each feature (each column) had zero mean and variance equal to 1: the normalization consists on evaluating the mean and the variance of each column. Each mean value (m) was subtracted from the corresponding column for the 36 patients then the result was divided by the standard deviation (s).

-	For the data_test the normalization was performed by the measured mean and the measured standard deviation of the training data.

- Define F0 as the target feature that we will estimate from the other features, y_train is the column vector that contains this Target feature. Knowing that both jitter% (column 6) and UPDRS (column 5) have been analyzed by the regression process but in this report, only UPDRS results were presented.

- y_train= data_train_norm[:,F0] and the matrix X_train is the data_train_norm without the column F0. The same thing for the test set.

Although, performing a regression technique to predict UPDRS is not too much efficient since not only Parkinson’s disease affect voice parameters.

## 3- MSE (Minimum square Error): 
This method consists on minimizing the square error between the original vector and the predicted one:

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq.PNG)

Where e (w) is a scalar and y is the original column vector with N elements, X is the matrix with N rows and 21 columns and w is an unknown column vector with 21 elements. The square error e (w) has just one minimum value which is ŵ that is obtained through calculating the ∇e(i) and set it to zero then solve the equation to get the following results:

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq%202.PNG)

## 4- Gradient Descent :
The previous method (MSE) requires the matrix to be inverted, which may be too complex in some cases (like image processing) then it is preferred to use the following method, which is an iterative solution: 

- Start with an initial guess of ŵ (0), which is a random vector of 21 random variables. 

- Evaluate the gradient:

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq3.PNG)

- Update the guess:  

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq4.PNG)

- Increase i by 1 and go back to step 2, until a stopping rule is satisfied 

•	An example of stopping rule is ||ŵ(i+1)-ŵ(i)||<∊, in this lab ∊ =10<sup>-8</sup>

•	In this lab γ was chosen 0.0001:

  o	If γ is large, we jump around the optimum value but we never reach it because the jump is too large. 
  o	If γ is, too small it takes a lot of time to reach the optimum value.
  
This method converges to the MSE solution if γ was chosen correctly.

## 5- Steepest Descent:
Same as before this algorithm is also an iterative one, which allows to get a faster convergence with respect to the previous one. It consists on finding the “optimum” value of γ at each step. Besides, in this approach need to start with a random vector ŵ (0) as before.

The algorithm has the following steps:

- Start from an initial guess ŵ (0)

- Evaluate the gradient and the Hessian matrix at point ŵ (i)

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq5.PNG)

- Find the new point as:

![alt text](https://github.com/BaddyMAK/Linear-regression-with-ML/blob/main/results/eq6.PNG)

- Set i: = i + 1, go back to step 2, unless a stop condition is met.



















