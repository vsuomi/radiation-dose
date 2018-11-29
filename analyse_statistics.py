# -*- coding: utf-8 -*-
'''
Created on Thu Nov 29 11:58:10 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This function is used to analyse the distribution of the target/feature
    
    Input:
        
        sample_data: Pandas dataframe of sample data
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

#%% define function

def analyse_statistics(sample_data):
    
    # plot Histogram

    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    sns.distplot(sample_data, fit = sp.stats.norm)
    plt.grid()
    
    # get the fitted parameters used by the function
    
    (mu, sigma) = sp.stats.norm.fit(sample_data)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Count')
    plt.xlabel('Target value')
    plt.title(sample_data.columns.values[0])
    
    plt.subplot(1, 2, 2)
    sp.stats.probplot(sample_data.T.squeeze(), plot = plt)
    plt.grid()
    
    print('Skewness: %f' % sample_data.skew())
    print('Kurtosis: %f' % sample_data.kurt())
    print('Mean: %.2f' % mu)
    print('Std: %.2f' % sigma)
    