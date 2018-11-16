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

def plot_regression_performance(history, training_targets, training_predictions, 
                                validation_targets, validation_predictions):
    
    # training error
    
    f1 = plt.figure(figsize = (18, 4))
    plt.subplot(1, 3, 1)
    plt.title('Training and validation error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label = 'Training')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
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
    training_error = training_predictions.as_matrix() - training_targets.as_matrix()
    validation_error = validation_predictions.as_matrix() - validation_targets.as_matrix()
    plt.hist(training_error, bins = 50, label = 'Training')
    plt.hist(validation_error, bins = 50, label = 'Validation')
    plt.grid()
    plt.legend()
    
    return f1


