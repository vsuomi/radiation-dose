# -*- coding: utf-8 -*-
'''
Created on Fri Nov 16 11:35:48 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This function is used for plotting the performance metrics from a trained
    Keras model
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import numpy as np

#%% define function

def plot_regression_performance(model, losses, training_targets, training_predictions, 
                                validation_targets, validation_predictions):
    
    # training error
    
    f1 = plt.figure(figsize = (18, 4))
    plt.subplot(1, 3, 1)
    plt.title('Training and validation error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    
    if model == 'keras':
        plt.plot(losses.epoch, np.array(losses.history['loss']),
                 label = 'Training')
        plt.plot(losses.epoch, np.array(losses.history['val_loss']),
                 label = 'Validation')
    if model == 'xgboost':
        plt.plot(np.array(losses['training']['rmse']),
                 label = 'Training')
        plt.plot(np.array(losses['validation']['rmse']),
                 label = 'Validation')
    plt.grid()
    plt.legend()
    
    # prediction accuracy
       
    plt.subplot(1, 3, 2)
    plt.title('Prediction accuracy')
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.scatter(training_targets, training_predictions, label = 'Training')
    plt.scatter(validation_targets, validation_predictions, label = 'Validation')
    plt.plot([0, 300], [0, 300], color = 'k')
    plt.grid()
    plt.legend()
    
    # prediction error
    
    plt.subplot(1, 3, 3)
    plt.title('Prediction error')
    plt.xlabel('Prediction error')
    plt.ylabel('Count')
    training_error = training_predictions.values - training_targets.values
    validation_error = validation_predictions.values - validation_targets.values
    plt.hist(training_error, bins = 50, label = 'Training')
    plt.hist(validation_error, bins = 50, label = 'Validation')
    plt.grid()
    plt.legend()
    
    return f1


