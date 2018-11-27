# -*- coding: utf-8 -*-
'''
Created on Fri Nov 16 09:36:50 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This model is used to predict radiation dose from pre-treatment patient 
    parameters
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import keras as k
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy as sp
import time
import os

from save_load_variables import save_load_variables
from plot_regression_performance import plot_regression_performance

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Radiation dose\radiation-dose\radiation_dose_data.csv', sep = ',')

#%% check for duplicates

duplicates = any(dataframe.duplicated())

#%% handle nan values

nan_percent = dataframe.isnull().mean() * 100
dataframe = dataframe.dropna(subset = ['paino'])
dataframe = dataframe.fillna(dataframe.median())

#%% display target histogram

dataframe['Korjattu_DAP_GYcm2'].hist(bins = 20)

#%% create synthetic features

dataframe['BSA'] = 0.007184 * dataframe['paino'].pow(0.425) * dataframe['pituus'].pow(0.725)

#%% calculate correlation matrix

corr_mat = dataframe.corr()

#%% define feature and target labels

feature_labels = ['BSA', 'Patient_sex', 'Age', 
                  'I20.81_I21.01_I21.11_or_I21.41', 'FN1AC', 'FN2BA',
                  'FN2AA', 'TFC00', 'n_tmp_1', 'n_tmp_2', 'n_tmp_3', 
                  'ind_pci_in_stemi', 'ind_flap_failure', 'ind_nstemi', 
                  'ind_diag', 'ind_uap', 'ind_heart_failure', 'ind_stemi_other',
                  'ind_stable_ap', 'ind_arrhythmia_settl', 'suonia_2_tai_yli', 
                  'lm_unprotected', 'Aiempi_ohitusleikkaus', 
                  'restenosis',
                  'IVUS', 'OCT']

#feature_labels = ['paino', 'pituus', 'Patient_sex', 'Age', 
#                  'I20.81_I21.01_I21.11_or_I21.41', 'FN1AC', 'FN2BA',
#                  'FN2AA', 'TFC00', 'n_tmp_1', 'n_tmp_2', 'n_tmp_3', 
#                  'ind_pci_in_stemi', 'ind_flap_failure', 'ind_nstemi', 
#                  'ind_diag', 'ind_uap', 'ind_heart_failure', 'ind_stemi_other',
#                  'ind_stable_ap', 'ind_arrhythmia_settl', 'suonia_2_tai_yli', 
#                  'lm_unprotected', 'Aiempi_ohitusleikkaus', 
#                  'restenosis',
#                  'add_stent_1', 'add_stent_2_tai_yli', 'sten_post_0', 
#                  'sten_post_25', 'sten_post_60', 'sten_post_85', 'sten_post_100',
#                  'sten_pre_100', 'sten_pre_85', 'sten_pre_60', 'AHA_a', 'AHA_b1',
#                  'AHA_b2', 'AHA_c', 'AHA_cto', 'IVUS', 'OCT']

#feature_labels = ['paino', 'pituus', 'Patient_sex', 'Age', 
#                  'I20.81_I21.01_I21.11_or_I21.41', 'FN1AC', 'FN2BA',
#                  'FN2AA', 'TFC00', 'n_tmp_1', 'n_tmp_2', 'n_tmp_3', 
#                  'ind_pci_in_stemi', 'ind_flap_failure', 'ind_nstemi', 
#                  'ind_diag', 'ind_uap', 'ind_heart_failure', 'ind_stemi_other',
#                  'ind_stable_ap', 'ind_arrhythmia_settl', 'suonia_2_tai_yli', 
#                  'lm_unprotected', 'Aiempi_ohitusleikkaus', 'im', 'lada', 
#                  'ladb', 'ladc', 'lcxa', 'lcxb', 'lcxc', 'ld1', 'ld2',
#                  'lm', 'lom1', 'lom2', 'lpl', 'rcaa', 'rcab',
#                  'rcac', 'rita', 'rpd', 'rpl', 'vgrca_ag', 'vglca1_ag', 
#                  'restenosis', 'stent_dimension', 'ball_dimension',
#                  'add_stent_1', 'add_stent_2_tai_yli', 'sten_post_0', 
#                  'sten_post_25', 'sten_post_60', 'sten_post_85', 'sten_post_100',
#                  'sten_pre_100', 'sten_pre_85', 'sten_pre_60', 'AHA_a', 'AHA_b1',
#                  'AHA_b2', 'AHA_c', 'AHA_cto', 'IVUS', 'OCT']

#feature_labels = ['paino', 'pituus', 'Patient_sex', 'Age', 
#                  'I20.81_I21.01_I21.11_or_I21.41', 'I35.0', 'FN1AC', 'FN2BA',
#                  'FN2AA', 'TFC00', 'n_tmp_1', 'n_tmp_2', 'n_tmp_3', 
#                  'ind_pci_in_stemi', 'ind_flap_failure', 'ind_nstemi', 
#                  'ind_diag', 'ind_uap', 'ind_heart_failure', 'ind_stemi_other',
#                  'ind_stable_ap', 'ind_arrhythmia_settl', 'suonia_2_tai_yli', 
#                  'lm_unprotected', 'Aiempi_ohitusleikkaus', 'im', 'lada', 
#                  'ladb', 'ladc', 'lcxa', 'lcxb', 'lcxc', 'ld1', 'ld2', 'lita',
#                  'lm', 'lom1', 'lom2', 'lpd', 'lpl', 'ram_rv', 'rcaa', 'rcab',
#                  'rcac', 'rita', 'rpd', 'rpl', 'vgrca_ag', 'vglca1_ag', 
#                  'vglca2_ag', 'restenosis', 'stent_dimension', 'ball_dimension',
#                  'add_stent_1', 'add_stent_2_tai_yli', 'sten_post_0', 
#                  'sten_post_25', 'sten_post_60', 'sten_post_85', 'sten_post_100',
#                  'sten_pre_100', 'sten_pre_85', 'sten_pre_60', 'AHA_a', 'AHA_b1',
#                  'AHA_b2', 'AHA_c', 'AHA_cto', 'IVUS', 'OCT']

target_label = ['Korjattu_DAP_GYcm2']

#%% extract features and targets

features = dataframe[feature_labels]
targets = dataframe[target_label]

#%% scale features

scaled_features = pd.DataFrame(sp.stats.mstats.zscore(features),
                               columns = list(features), 
                               index = features.index, dtype = float)

#%% combine dataframes

concat_dataframe = pd.concat([scaled_features, targets], axis = 1)

#%% randomise and divive data for cross-validation

split_ratio = 0.2
training_set, holdout_set = train_test_split(concat_dataframe, test_size = split_ratio)
validation_set, testing_set = train_test_split(holdout_set, test_size = 0.5)

#%% define features and targets

training_features = training_set[feature_labels]
validation_features = validation_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
validation_targets = validation_set[target_label]
testing_targets = testing_set[target_label]

#%% build and train neural network model

# define parameters

learning_rate = 0.001
n_epochs = 100
n_neurons = 64
n_layers = 1
batch_size = 5
l1_reg = 0.5
l2_reg = 0.1
batch_norm = False
dropout = None

# build model

if 'model' in locals():
    del model

model = k.models.Sequential()

model.add(k.layers.Dense(n_neurons, 
                         input_shape = (training_features.shape[1],),
                         kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                         activation = 'relu'))
if batch_norm is True:
    model.add(k.layers.BatchNormalization())
if dropout is not None:
    model.add(k.layers.Dropout(dropout))

i = 1   
while i < n_layers:
    model.add(k.layers.Dense(n_neurons,
                             kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                             activation = 'relu'))
    if batch_norm is True:
        model.add(k.layers.BatchNormalization())
    if dropout is not None:
        model.add(k.layers.Dropout(dropout))
    i += 1

model.add(k.layers.Dense(1))

model.compile(optimizer = k.optimizers.Adam(lr = learning_rate),
              loss = 'mean_squared_error',
              metrics = ['mean_absolute_error'])

model.summary()

# train model

class PrintDot(k.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end = '')
    
timestr = time.strftime('%Y%m%d-%H%M%S')

history = model.fit(training_features, training_targets, verbose = 0, callbacks = [PrintDot()],
                    batch_size = batch_size, epochs = n_epochs,
                    validation_data = (validation_features, validation_targets))

#%% evaluate model performance

# calculate loss metrics

training_loss, training_error = model.evaluate(training_features, training_targets)
validation_loss, validation_error = model.evaluate(validation_features, validation_targets)

# make predictions

training_predictions = model.predict(training_features)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

validation_predictions = model.predict(validation_features)
validation_predictions = pd.DataFrame(validation_predictions, columns = target_label,
                                      index = validation_features.index, dtype = float)

# plot training performance

f1 = plot_regression_performance(history, training_targets, training_predictions, 
                                 validation_targets, validation_predictions)

#%% save model

model_dir = 'Keras models\\%s_TE%d_VE%d' % (timestr, 
                                            round(training_error), 
                                            round(validation_error))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'evaluation_metrics.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'learning_rate': learning_rate,
                     'n_epochs': n_epochs,
                     'n_neurons': n_neurons,
                     'n_layers': n_layers,
                     'batch_size': batch_size,
                     'l1_reg': l1_reg,
                     'l2_reg': l2_reg,
                     'batch_norm': batch_norm,
                     'dropout': dropout,
                     'nan_percent': nan_percent,
                     'duplicates': duplicates,
                     'corr_mat': corr_mat,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'history': history,
                     'model_dir': model_dir,
                     'dataframe': dataframe,
                     'concat_dataframe': concat_dataframe,
                     'holdout_set': holdout_set,
                     'training_set': training_set,
                     'training_features': training_features,
                     'training_targets': training_targets,
                     'validation_set': validation_set,
                     'validation_features': validation_features,
                     'validation_targets': validation_targets,
                     'testing_set': testing_set,
                     'testing_features': testing_features,
                     'testing_targets': testing_targets,
                     'scaled_features': scaled_features,
                     'features': features,
                     'targets': targets,
                     'feature_labels': feature_labels,
                     'target_label': target_label}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')

model.save(model_dir + '\\' + 'keras_model.h5')


