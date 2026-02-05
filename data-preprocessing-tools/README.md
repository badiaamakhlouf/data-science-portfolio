# Welcome to Feature Engineering and Feature Extraction

Feature engineering is a crucial step in turning raw data into consistent, coherent and relevant set of features for a high-performing model building.
It is a a preliminary step before model training phase and it must be accurate to enable shaping and selecting the right data to create meaningful features for better insights detection and model robustness.

During this article, we will delve into the intricacies of feature engineering and feature extraction. From the fundamentals of manipulating data to the art of selecting and creating features, you'll gain a comprehensive understanding of how to optimize your data for machine learning success.

- Are you ready to unlock the true potential of your data? Let's get started! 🚀🎯
  
## Contents
 - Introduction to Feature Engineering
 - Handling Missing Data
 - Dealing with Categorical Data
 - Scaling and Normalization
 - Feature Transformation 
 - Feature Selection
 - Feature Extraction
 - Advanced Feature Engineering
 - Automated Feature Engineering
 - Conclusion

<div>
<img src="images/feature_engineering.jpeg "Feature engineering" width="500"/>
</div>

*Source : [springboard.com](https://www.springboard.com/blog/data-science/how-to-become-data-engineer/)*
  
## 1- Introduction to Feature Engineering
 Feature engineering refers to the process of raw data manipulation such as addition, deletion, combination, mutation etc. It encompasses the process of creating new features or modifying existing ones to improve the performance of a machine learning model.

 In addition, feature engineering in Machine Learning involves extracting useful features from the given input data to better learn the target using our choosen machine learning model. It involves transforming data to forms that better relate to the underlying target to be learned. 
 At the end, only the most significant features, that have linear relation with the target, are given to the model and the remaining features are discarded.
 Feature engineering encompasses several crucial phases to enhance the performance and interpretability of machine learning models:

## 2- Handling Missing Data
## 3- Dealing with Categorical Data
- Encoding Categorical Variables: converting categorical variables into numerical representations that can be used by machine learning algorithms.
  - Feature encoding : Encoding categorical values to numeric variables
    - Solution 1: Using Python’s Category Encoder Library
    - Solution 2: Dummy Variable Encoding
    - Solution 3: Using Scikit-learn
    - Solution 4 : Ordinal Encoding - Manual
- Feature encoding : transforming categorical values to numeric variables : both nominal and ordinal data were considered. I have used :
    - Python’s Category Encoder Library :
      - One-hot Encoding: converts variables that take multiple values into binary (0,1) variables one for each category. This creates several new variables.
      - Label Encoding
      - Ordinal Encoding: converting ordered categories to numerical values, assigning each categorical value to a number such as 0,1,2 etc. while respecting the order : for example LOW==> 0, Medium==>1, High==>2.
      - Helmert Encoding
      - Binary Encoding : convert variables to either 0 or 1 suitable for variables with two values such as `YES/NO`, `TRUE/FALSE`
      - Frequency Encoding
      - Mean Encoding
      - Weight of Evidence Encoding
      - Probability Ratio Encoding
      - Hashing Encoding
      - Backward Difference Encoding
      - Leave One Out Encoding
      - James-Stein Encoding
      - M-estimator Encoding
      - Thermometer Encoder
    - Dummy Variable Encoding using `pd.get_dummies()`
    - Using Scikit-learn such as `OneHotEncoder`
    - Ordinal Encoding such as `cat.codes` of pandas
    - How to choose the best Encoding Method
   
## 4- Scaling and Normalization
- Scaling and Standardization: adjusting the scale of features to a similar range or standardizing them to have a mean of 0 and a standard deviation of 1.
- Normalization: scaling features to a range between 0 and 1.
-  - Normalization : 
    - Min-Max feature scaling (Normalization) using `MinMaxScaler`
    - Maximum absolute scaling (Normalization) using `.abs().max()`
  - Standardisation
    - Z-score method (Standardization Scaling) using `mean()` and `std()` or `StandardScaler`
  - - Feature Scaling
     - Data Normalization
       - Min-Max feature scaling (Normalization
       - Maximum absolute scaling (Normalization)
     - Standard scaling
        - Z-score method
        - Data Standardisation using StandardScaler
  - Robust Scaling
  - log scaling
    
## 5- Feature Transformation 
Data transformation is indeed one subtask within the broader field of feature engineering in machine learning. It is a specific aspect of feature engineering that involves modifying the raw data to make it more suitable for the learning algorithm. 

Moreover, The data transformation step could be required either before some data cleaning tasks or between data cleaning and Exploratory Data Analysis (EDA). Data transformation before data cleaning, is the process of converting data from one format to another such as from JSON to CSV or data aggregation etc. After data cleaning, this process helps convert our cleaned data into useful information and more significant features. 

Data transformation can include various operations such as:

- Feature Extraction: reducing dimensionality through techniques like Principal Component Analysis (PCA) or extracting meaningful features from raw data.
  
Effective data transformation can have a significant impact on the performance of machine learning models by improving their ability to extract patterns and relationships from the data. It is an essential step in the overall feature engineering process, which aims to enhance the representational power of the features used by the model.

During the data transformation part, I have accomplished the next steps: 

- Data transformation : means transforming features to be on a similar scale which improves the performance and training stability of the model. I have used :
 
  - Log transformation
    - Log scaling or Log transformation using `np.log()`
- Logarithmic Transformations: applying logarithmic, square root, or other power transformations to handle skewed distributions.
- Binning or Discretization: arouping continuous data into bins or discrete intervals.
- Creating Interaction Terms: combining existing features to capture interactions between them.
- Data Transformation: In certain cases, it is essential to transform variables before incorporating them into the model. This transformation can involve scaling, normalization, or encoding categorical variables to ensure compatibility and improved model accuracy.
- Log transformation
- Polynomial transformation

## 6- Feature Selection
- Feature Selection: This involves the strategic process of choosing a subset of features to include in the model. By selecting the most relevant features, we aim to optimize model performance and reduce complexity.

## 7- Feature Extraction
- Feature extraction
  - Principal Component Analysis (PCA)
  - Singular Value Decomposition (SVD)
    
## 8- Advanced Feature Engineering
- Text Data Processing: For tasks involving text data, a critical step is to tokenize, vectorize, and preprocess the text. Tokenization breaks down text into meaningful units, vectorization converts text into numerical format, and preprocessing enhances the text's suitability for machine learning models.

- Time-Series Feature Engineering: When dealing with time-series data, additional considerations come into play. This phase involves creating lag features, calculating rolling statistics, or generating other time-related features. These enhancements provide the model with insights into temporal patterns, enabling more effective analysis and predictions in time-series scenarios.

## 9- Automated Feature Engineering 

## 10- Conclusion
Through these phases, feature engineering plays a pivotal role in shaping input data to empower machine learning models with the information needed to make accurate and meaningful predictions. In order to achieve an effective feature engineering, it is important to understand deeply the business problem and the given data via perforing exploratory data analysis and data cleaning.

- Variable transformation and feature engineering are integral components of the machine learning pipeline.
- Transformed data is easier to understand and to analyze either by computer or human.
- Feature engineering involves creating new features, selecting relevant ones, and extracting meaningful information to improve a model's ability to capture patterns and relationships within the data.
- Effective variable transformation and feature engineering require a deep understanding of the data and domain knowledge.
