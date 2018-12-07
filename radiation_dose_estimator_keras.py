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
from analyse_statistics import analyse_statistics
from analyse_correlation import analyse_correlation
from analyse_feature_correlation import analyse_feature_correlation

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Radiation dose\radiation-dose\radiation_dose_data.csv', sep = ',')

#%% check for duplicates

duplicates = any(dataframe.duplicated())

#%% calculate correlation and standard deviation matrices

std_mat, corr_mat, most_corr = analyse_correlation(dataframe, 10, 'Korjattu_DAP_GYcm2')

#%% handle nan values

# calculate nan percent for each label

nan_percent = pd.DataFrame(dataframe.isnull().mean() * 100, columns = ['% of NaN'])

# drop nan values

#dataframe = dataframe.dropna()
dataframe = dataframe.dropna(subset = ['paino'])
#dataframe = dataframe.dropna(subset = ['pituus'])

# fill in mutually exclusive categorical values

sten_post = dataframe[['sten_post_0', 'sten_post_25', 'sten_post_60', 
                       'sten_post_85', 'sten_post_100']].idxmax(axis = 1)
sten_post = sten_post.fillna(sten_post.mode()[0])
sten_post = pd.get_dummies(sten_post).astype(int)
sten_post = sten_post[['sten_post_0', 'sten_post_25', 'sten_post_60', 
                       'sten_post_85', 'sten_post_100']]

sten_pre = dataframe[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']].idxmax(axis = 1)
sten_pre = sten_pre.fillna(sten_pre.mode()[0])
sten_pre = pd.get_dummies(sten_pre).astype(int)
sten_pre = sten_pre[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']]

AHA = dataframe[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']].idxmax(axis = 1)
AHA = AHA.fillna(AHA.mode()[0])
AHA = pd.get_dummies(AHA).astype(int)
AHA = AHA[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']]

# replace missing values

dataframe['paino'] = dataframe['paino'].fillna(dataframe['paino'].mean())
dataframe['pituus'] = dataframe['pituus'].fillna(dataframe['pituus'].mean())
dataframe['ind_pci_in_stemi'] = dataframe['ind_pci_in_stemi'].fillna(dataframe['ind_pci_in_stemi'].mode()[0])
dataframe['ind_flap_failure'] = dataframe['ind_flap_failure'].fillna(dataframe['ind_flap_failure'].mode()[0])
dataframe['ind_nstemi'] = dataframe['ind_nstemi'].fillna(dataframe['ind_nstemi'].mode()[0])
dataframe['ind_diag'] = dataframe['ind_diag'].fillna(dataframe['ind_diag'].mode()[0])
dataframe['ind_uap'] = dataframe['ind_uap'].fillna(dataframe['ind_uap'].mode()[0])
dataframe['ind_heart_failure'] = dataframe['ind_heart_failure'].fillna(dataframe['ind_heart_failure'].mode()[0])
dataframe['ind_stemi_other'] = dataframe['ind_stemi_other'].fillna(dataframe['ind_stemi_other'].mode()[0])
dataframe['ind_stable_ap'] = dataframe['ind_stable_ap'].fillna(dataframe['ind_stable_ap'].mode()[0])
dataframe['ind_arrhythmia_settl'] = dataframe['ind_arrhythmia_settl'].fillna(dataframe['ind_arrhythmia_settl'].mode()[0])
dataframe['suonia_2_tai_yli'] = dataframe['suonia_2_tai_yli'].fillna(dataframe['suonia_2_tai_yli'].mode()[0])
dataframe['lm_unprotected'] = dataframe['lm_unprotected'].fillna(dataframe['lm_unprotected'].mode()[0])
dataframe['im'] = dataframe['im'].fillna(dataframe['im'].mode()[0])
dataframe['lada'] = dataframe['lada'].fillna(dataframe['lada'].mode()[0])
dataframe['ladb'] = dataframe['ladb'].fillna(dataframe['ladb'].mode()[0])
dataframe['ladc'] = dataframe['ladc'].fillna(dataframe['ladc'].mode()[0])
dataframe['lcxa'] = dataframe['lcxa'].fillna(dataframe['lcxa'].mode()[0])
dataframe['lcxb'] = dataframe['lcxb'].fillna(dataframe['lcxb'].mode()[0])
dataframe['lcxc'] = dataframe['lcxc'].fillna(dataframe['lcxc'].mode()[0])
dataframe['ld1'] = dataframe['ld1'].fillna(dataframe['ld1'].mode()[0])
dataframe['ld2'] = dataframe['ld2'].fillna(dataframe['ld2'].mode()[0])
dataframe['lita'] = dataframe['lita'].fillna(dataframe['lita'].mode()[0])
dataframe['lm'] = dataframe['lm'].fillna(dataframe['lm'].mode()[0])
dataframe['lom1'] = dataframe['lom1'].fillna(dataframe['lom1'].mode()[0])
dataframe['lom2'] = dataframe['lom2'].fillna(dataframe['lom2'].mode()[0])
dataframe['lpd'] = dataframe['lpd'].fillna(dataframe['lpd'].mode()[0])
dataframe['lpl'] = dataframe['lpl'].fillna(dataframe['lpl'].mode()[0])
dataframe['ram_rv'] = dataframe['ram_rv'].fillna(dataframe['ram_rv'].mode()[0])
dataframe['rcaa'] = dataframe['rcaa'].fillna(dataframe['rcaa'].mode()[0])
dataframe['rcab'] = dataframe['rcab'].fillna(dataframe['rcab'].mode()[0])
dataframe['rcac'] = dataframe['rcac'].fillna(dataframe['rcac'].mode()[0])
dataframe['rita'] = dataframe['rita'].fillna(dataframe['rita'].mode()[0])
dataframe['rpd'] = dataframe['rpd'].fillna(dataframe['rpd'].mode()[0])
dataframe['rpl'] = dataframe['rpl'].fillna(dataframe['rpl'].mode()[0])
dataframe['vgrca_ag'] = dataframe['vgrca_ag'].fillna(dataframe['vgrca_ag'].mode()[0])
dataframe['vglca1_ag'] = dataframe['vglca1_ag'].fillna(dataframe['vglca1_ag'].mode()[0])
dataframe['vglca2_ag'] = dataframe['vglca2_ag'].fillna(dataframe['vglca2_ag'].mode()[0])
dataframe['restenosis'] = dataframe['restenosis'].fillna(dataframe['restenosis'].mode()[0])
dataframe['stent_dimension'] = dataframe['stent_dimension'].fillna(dataframe['stent_dimension'].mean())
dataframe['ball_dimension'] = dataframe['ball_dimension'].fillna(dataframe['ball_dimension'].mean())
dataframe['add_stent_1'] = dataframe['add_stent_1'].fillna(0)
dataframe['add_stent_2_tai_yli'] = dataframe['add_stent_2_tai_yli'].fillna(0)
dataframe[['sten_post_0', 'sten_post_25', 'sten_post_60', 'sten_post_85', 'sten_post_100']] = sten_post
dataframe[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']] = sten_pre
dataframe[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']] = AHA

# check for nan values

dataframe.isnull().values.any()

#%% create synthetic features

#dataframe['BSA'] = 0.007184 * dataframe['paino'].pow(0.425) * dataframe['pituus'].pow(0.725)

#%% analyse individual feature correlations

analyse_feature_correlation(dataframe, 'paino', 'Korjattu_DAP_GYcm2', False)
analyse_feature_correlation(dataframe, 'AHA_cto', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'Patient_sex', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'FN2BA', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'I20.81_I21.01_I21.11_or_I21.41', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'add_stent_2_tai_yli', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'n_tmp_3', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'sten_post_100', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(dataframe, 'suonia_2_tai_yli', 'Korjattu_DAP_GYcm2', True)

#%% define feature and target labels

feature_labels = ['paino', 'AHA_cto', 'Patient_sex', 'FN2BA',
                  'I20.81_I21.01_I21.11_or_I21.41', 'add_stent_2_tai_yli',
                  'n_tmp_3', 'sten_post_100', 'suonia_2_tai_yli',]

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

#%% analyse target

analyse_statistics(targets)

#%% scale features

# z-score

#scaled_features = pd.DataFrame(sp.stats.mstats.zscore(features),
#                               columns = list(features), 
#                               index = features.index, dtype = float)

# log (for skewed data)

scaled_features = np.log1p(features)

# box cox (for skewed data)

#scaled_features = sp.special.boxcox1p(features, 0.15)

#%% log transform targets (for skewed data)

scaled_targets = np.log1p(targets)

#%% combine dataframes

concat_dataframe = pd.concat([scaled_features, scaled_targets], axis = 1)

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
n_layers = 2
batch_size = 5
l1_reg = 0.0
l2_reg = 0.001
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

# convert targets to linear units (for skewed data)

training_targets_lin = np.exp(training_targets) - 1
validation_targets_lin = np.exp(validation_targets) - 1

training_predictions_lin = np.exp(training_predictions) - 1
validation_predictions_lin = np.exp(validation_predictions) - 1

# plot training performance

f1 = plot_regression_performance(history, training_targets_lin, training_predictions_lin, 
                                 validation_targets_lin, validation_predictions_lin)

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
                     'most_corr': most_corr,
                     'corr_mat': corr_mat,
                     'std_mat': std_mat,
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


