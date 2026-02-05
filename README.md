# Data Cleaning, Preprocessing and EDA

In this article, I am going to present you all important points you need to know and to master, as a data scientist,
about Data Preprocessing step mainly data cleaning task.
Choosing the correct and the consistent data values helps build a robust machine learning model, finding proper insights or trends and so taking the right decision.



## Contents 

1. Short summary
2. Data Reading
3. Data Cleaning
4. Exploratory Data Analysis
5. Conclusion 
  



## 1. Summary 

As a data scientist, working directly with raw data without performing any cleaning or transformation tasks could be useless and critical. Firstly, raw data is unclean, incoherent and poorly structured. Secondly, it may contain duplicates, missing and outdated values. Finally, computers cannot intuitively process raw data like a human mind can. 

Thus, if the data quality is not good enough, we will not be able to build a reliable model and then your project could fail easily. 

## 2. Data Reading 

Retrieving Data can be accomplished using multiple data sources such as: 

- Databases SQL/NOSQL : MySQL, Cassandra etc.
- Data lakes : Snowflake, S3 etc.
- APIs
- Cloud data sources
  
SQL databases uses structured data that is coming under the .csv and .tsv formats which are the most common formats for delimited data flat files. Sometimes, you can use special characters as separators such as white space or normal letter like 't', 'v' etc. However, NOSQL databases and APIs uses unstructured data stored under .JSON format, JSON corresponds to JavaScript Object Notation.

Another option for retrieving data could be the Parquet format using `read_parquet()` of our famous `pandas` library but it is not recommended to use it as it requires very high CPU usage.

## 3. Data Cleaning


Data cleaning is the initial step that should be accomplished within any data science project. 
It is an essential task that plays a significant role to improve raw data quality, to increase data coherence and accuracy 
and to ensure that findings are based on reliable information. 


Finally, it is important to evaluate how data cleaning has affected the results and conclusions of our data science project.
For example, to emphasise the role of data cleaning, we can compare the performance and accuracy of our machine learning model before and after cleaning. 

<p align="center">
<img src="images/Data-Cleaning-scaled.jpeg "Data cleaning" width="300"/>
</p>

Data cleaning is the initial step that each data scientist must apply before starting any analysis or investigations and of course before applying any machine learning algorithm. It is a procedure that helps determine duplicate, missing, inconsistent samples and remove them. It improves data accuracy and increases data quality via discarding invalid and unwanted information.

Data cleaning is the process of determining and fixing incorrect data. It can be in incorrect format, duplicates, corrupt, inaccurate, incomplete, or irrelevant etc.

In my notebook `data_cleaning_and_transformation` I have chosen one random dataset from Kaggle site :
[weather-dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset). This dataset conatins 12 columns and 96453 rows.

The 12 features are the next : `Formatted Date`, `Summary`, `Precip Type`, `Temperature (C)`, `Apparent Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, `Wind Bearing (degrees)`, `Visibility (km)`, `Loud Cover`, `Pressure (millibars)`, `Daily Summary`. I have selected some of them for my accomplished tasks.  

During the data cleaning phase, I had performed the next steps: 

- How to detect and Handle **Duplicates** and removing them.
- How to detect and Handle **missing values (NaN)** : removing or impute them using KNNImputer, mean or median etc.
- How to deal with **Categorical data** : nominal and ordinal examples.
- How to detect and handle **Outliers** : performing some plots and visualisations together with Interquartile range, Z-Score and 99th-percentile

Note : I have used various methods for the same tasks just to present you all and then you can choose whatever you want to apply in your code. 



## 4. Exploratory Data Analysis (EDA)
EDA helps identify most relevant features among the existing ones and it helps determine next steps to perform before applying any type of machine learning algorithm.
Next steps could be related to data transformation (aggregation, log transformation, normalisation etc.), or it could be related to increasing the size of input data set via exploring new data sources.
Briefly, EDA helps data scientists decide what they exactly need to reach their target goal and find the required answers and to make it easy to discover patterns, trends, anomalies etc. 

In our Jupyter notebook we had accomplished the next steps, respectively : 
- Import the required libraries
- Data reading
- Data exploration
    -  More information: `.info()`
    -  Data frame shape: `.shape`
    -  Columns names : useful for filtering data with columns values or selecting a subset of data `.names`
    -  Statistical summary : `.describe()`
    -  Values Count
    -  Data sampling
    -  Data Grouping using `groupby` 
    -  Data filtering : by single or by multiple conditions using logical operatore & (and), |(or).
-  Data analysis
     - Univariate Analysis : distplot and histplot
     - Bivariate Analysis
     - Multivariate Analysis
       - Correlation matrix and heatmap
-  Data visualization
     - `boxplot`
     - `histplot`
     - Scatter plot
 
In addition, we can go deeper with data wrangling and perform more exploration via transforming some variables such as Changing to datetime format using `pd.to_datetime(data['DATE'], format='%b-%y')` etc.


## 5. Conclusion 

• Raw data could be challenging to work with as it could be skewed and it has lot of defects.

• Multiple sources could be used to read data from such as databases, APIs, datalakes etc.

• Data cleaning returns high quality data which increases overall productivity, building valid model and simplify taking right decisions. 


# Chronic kidney disease classification by Decision tree - Binary_Classification
## Objectives
This work aims to perform binary classification of chronic kidney disease using the C4.5 algorithm of scikit-learn. Here I need to estimate the last column from the remained features using the Scikit Learn Python library that implements C4.5 or similar algorithm for hierarchical classification.

The C4.5 algorithm allows the generation of a decision tree based on the normalized mutual information or what is also called normalized information gain as a classification criteria to assign the convenient attribute at each node of the tree and to classify patients. The generated decision tree is easily read and understood.

Besides, the Scikit Learn implementation requires numerical data only so all categorical features must be mapped into numbers.
•	Yes to 1 and no to 0, Normal to 1 and abnormalto 0, etc.
## Dataset description:
The first 29 rows of the provided dataset file for this lab contain the description of the features and they must be skipped. The total dataset contains 400 instances classified into two classes as follow: 250 corresponding to the CKD and 150 not CKD.

From the provided description, it is noticed the existence of 24 features plus the last field (column 25) that corresponds to the class which specifies if the disease is present (ckd) or not (notckd). In the 24 features, it exists 11 numerical features and the rest are categorical or nominal features. 

Before the analyzing phase, the dataset must be prepared and cleaned due to the existence of some errors and missing fields. This task is usually needed and sometimes requires a long period. Moreover, the dataset contains some rows with an extra separator field “,”, so 26 columns were read instead of 25. The missing fields were identified by“?” and at the end the categorical features must be transformed into numerical but before that there are “hidden” typing error corresponds to typing “ yes” and not “yes” must be deleted.
Two options are exist for cleaning the data: manually by editing the original CSV file but it is preferable if the file is short while in our case it is better to exploit arguments of pandas.

In order to manage the NaN values the following two approaches were applied:
•	Removing the rows containing NaN values using the methods dropna of Pandas. 
•	Treating NaN values as another possible random variable but must be substituted with a number not already presented in the dataset. In this report -3 was chosen as random value.
## The Chronic kidney disease (CKD) (it was mentioned in my clustering repository also ):
A disease that affects the Kidney’s functionality. The kidney may loses its function for a period of months or years. This kind of disease has many causes where the major one is diabetes, high blood pressure, glomerulonephritis and polycystic kidney disease. Diagnosis is generally by blood tests to measure the glomerular filtration rate and urine tests to measure albumin.

## Results and discussion
The next Figures were obtained after executing my python script in spyder IDE:

###### Figure 1: Case1 : Removing the rows containing NaN values
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/Case%201.png)

###### Figure 2: Case2 : Substituting NaN value with -3 (random value)
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/Case%202.png)


The above classification trees were obtained by graphviz and saved as dot files. Then those dot files were converted into png by executing the next instruction in my command line : 

!dot -Tpng tree.dot -o tree.png 

The Figure 2 demonstrates how substituting the missing values with random number has given a larger and a more complex decision tree while removing the lines containing miss-ing data has given a simpler tree as shown in Figure 1.

This is due to the higher number of rows in case 2 (Figure 2), which comports 400 samples, but in the case 1 (Figure 1), the number of samples was only 157. The most important feature for classification in case 1 (Figure 1) was the “al” (albumin). In case 2 (Figure 2), the added “-3” value had influenced the hierarchy of the tree where the most important features in classification became as follow: the “hemo” is the most important followed by the “sg” (specific gravity) at second rank and “al” (albumin) third rank. 
Moreover, one of the thresholds for the “al” feature has become negative (al <-1.5) means the -3 was combined with one of the classification attributes.

The two figures: 3 and 4 have illustrated the importance of each features in the dataset for both cases “Remove” and “Substitution” confirming the interpretation of the two above de-cision trees (Figure 2 and Figure 1).  In case 1, the most important feature is number 3, which is the “al” means the classification of all patients was based on only one feature while for the case 2 the most important feature is the “hemo” which is the feature number 14 fol-lowed by “sg” ( feature 2) and “al” (feature 3).

###### Figure 3: Features importance in case of substitution
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/feature%20importance%20substitution.png)

###### Figure 4: Features importance in case of remove
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/feature%20importance%20remove.png)









