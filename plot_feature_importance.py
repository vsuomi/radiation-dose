# -*- coding: utf-8 -*-
'''
Created on Fri Dec 14 14:33:05 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This function is used for plotting the feature importance
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import shap

#%% define function

def plot_feature_importance(model, training_features):
    
    f = plt.figure()
    shap_values = shap.TreeExplainer(model).shap_values(training_features)
    shap.summary_plot(shap_values, training_features)
    
    return f
