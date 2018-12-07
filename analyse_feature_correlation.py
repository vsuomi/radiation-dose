# -*- coding: utf-8 -*-
'''
Created on Fri Dec  7 13:34:34 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    December 2018
    
@description:
    
    This function is used to analyse feature correlation with the target
    
    input:
        
        dataframe - Pandas dataframe of features and target
        feature - string specifying the feature variable
        target - string specifying the target variable
        categorical - True or False if feature is catecorical
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% define function

def analyse_feature_correlation(dataframe, feature, target, categorical):
    
    if categorical:
        
        plt.figure(figsize = (6, 4))
        data = pd.concat([dataframe[target], dataframe[feature]], axis = 1)
        sns.boxplot(x = feature, y = target, data = data)
        
    else:
    
        plt.figure(figsize = (6, 4))
        sns.jointplot(x = dataframe[feature], y = dataframe[target], kind = 'reg')