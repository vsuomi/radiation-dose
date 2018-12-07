# -*- coding: utf-8 -*-
'''
Created on Thu Nov 29 13:56:01 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This function is used to calculate the most correlated features with the 
    target
    
    Input:
        
        dataframe: Pandas dataframe of features and target
        n: the number of most correlated features (int)
        target_label: list of target label
    
'''

#%% import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% define function

def analyse_correlation(dataframe, n, target_label):
    
    # calcualte correlation and standard deviation matrices
    
    corr_mat = dataframe.corr()
    std_mat = pd.DataFrame(dataframe.std(), columns = ['STD'])
    
    # most n correlated features
    
    cols = corr_mat.nlargest(n, target_label)[target_label].index
    
    # correlation matrix of n features
    
    corr_mat_n = dataframe[cols].corr()
    
    # display correlation matrix
    sns.set(font_scale = 1.25)
    sns.heatmap(corr_mat_n, cbar = True, annot = True, square = True, fmt = '.2f', 
                annot_kws = {'size': 10}, yticklabels = cols.values, 
                xticklabels = cols.values)
    plt.show()
    
    # return dataframe of most correlated features
    
    most_corr = pd.DataFrame(cols, columns = ['Most correlated features'])
    
    return std_mat, corr_mat, most_corr