# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:07:06 2018

@author: Badiaa
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


lis= ["age", "bp", "sg", "al", "su", "rbc", "pc",  "pcc", "ba", "bgr", "bu", 
      "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad",
      "appet", "pe", "ane", "class"]
xx = pd.read_csv('chronic_kidney_disease.arff', sep=',', skiprows=29, header=None, na_values=['?', '\t?'],
                  names = lis)
key_list =["normal", "abnormal", "present", "notpresent", "yes", "no", "poor", "good", "ckd", "notckd","ckd\t","\tyes", "\tnotckd",  "\tno" , ]

key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]

x= xx.copy()
x=x.replace(key_list,  key_val)

#%% removing NaN rows
data_frame1=x.dropna()
data_frame1.info()
Z= linkage(data_frame1, 'ward')

plt.figure(figsize=(25,10))
plt.xlabel('sample index')
plt.ylabel('distance')
plt.title('Hierarchical Clustering Dendrogram case "Remove"')
dendrogram(
        Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()  
#%% Substituting NaN value by -3
data_frame2=x.replace(np.nan, -3)
data_frame2.info()
Z1= linkage(data_frame2, 'ward')

plt.figure(figsize=(25,10))
plt.xlabel('sample index')
plt.ylabel('distance')
plt.title('Hierarchical Clustering Dendrogram case "Substitution"')
dendrogram(
        Z1, leaf_rotation=90., leaf_font_size=8.)
plt.show()  
