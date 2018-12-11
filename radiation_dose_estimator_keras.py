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

df = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Radiation dose\radiation-dose\radiation_dose_data.csv', sep = ',')
df_orig = df.copy()

#%% check for duplicates

duplicates = any(df.duplicated())

#%% create synthetic features

df['BSA'] = 0.007184 * df['paino'].pow(0.425) * df['pituus'].pow(0.725)

#%% calculate nan percent for each label

nan_percent = pd.DataFrame(df.isnull().mean() * 100, columns = ['% of NaN'])

# drop nan values

#df = df.dropna()
#df = df.dropna(subset = ['paino'])
#df = df.dropna(subset = ['AHA_cto'])
#df = df.dropna(subset = ['Patient_sex'])
#df = df.dropna(subset = ['FN2BA'])
#df = df.dropna(subset = ['I20.81_I21.01_I21.11_or_I21.41'])
#df = df.dropna(subset = ['add_stent_2_tai_yli'])
#df = df.dropna(subset = ['n_tmp_3'])
#df = df.dropna(subset = ['sten_post_100'])
#df = df.dropna(subset = ['suonia_2_tai_yli'])
#df = df.dropna(subset = ['pituus'])

#%% randomise and divive data for cross-validation

# split data

split_ratio = 0.2
training_set, holdout_set = train_test_split(df, test_size = split_ratio)
validation_set, testing_set = train_test_split(holdout_set, test_size = 0.5)

# obtain sizes

n_training = training_set.shape[0]
n_validation = validation_set.shape[0]
n_testing = testing_set.shape[0]

#%% calculate correlation and standard deviation matrices

std_mat, corr_mat, most_corr = analyse_correlation(training_set, 11, 'Korjattu_DAP_GYcm2')

#%% analyse individual feature correlations

analyse_feature_correlation(training_set, 'paino', 'Korjattu_DAP_GYcm2', False)
analyse_feature_correlation(training_set, 'AHA_cto', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'Patient_sex', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'FN2BA', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'I20.81_I21.01_I21.11_or_I21.41', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'add_stent_2_tai_yli', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'n_tmp_3', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'sten_post_100', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'suonia_2_tai_yli', 'Korjattu_DAP_GYcm2', True)
analyse_feature_correlation(training_set, 'pituus', 'Korjattu_DAP_GYcm2', False)

#%% analyse target

analyse_statistics(training_set[['Korjattu_DAP_GYcm2']])

#%% replace missing values in all datasets

# create dictionary for impute values based only on training data

impute_values = {'BSA': training_set['BSA'].mean(),
                 'paino': training_set['paino'].mean(),
                 'pituus': training_set['pituus'].mean(),
                 'ind_pci_in_stemi': training_set['ind_pci_in_stemi'].mode()[0],
                 'ind_flap_failure': training_set['ind_flap_failure'].mode()[0],
                 'ind_nstemi': training_set['ind_nstemi'].mode()[0],
                 'ind_diag': training_set['ind_diag'].mode()[0],
                 'ind_uap': training_set['ind_uap'].mode()[0],
                 'ind_heart_failure': training_set['ind_heart_failure'].mode()[0],
                 'ind_stemi_other': training_set['ind_stemi_other'].mode()[0],
                 'ind_stable_ap': training_set['ind_stable_ap'].mode()[0],
                 'ind_arrhythmia_settl': training_set['ind_arrhythmia_settl'].mode()[0],
                 'suonia_2_tai_yli': training_set['suonia_2_tai_yli'].mode()[0],
                 'lm_unprotected': training_set['lm_unprotected'].mode()[0],
                 'im': training_set['im'].mode()[0],
                 'lada': training_set['lada'].mode()[0],
                 'ladb': training_set['ladb'].mode()[0],
                 'ladc': training_set['ladc'].mode()[0],
                 'lcxa': training_set['lcxa'].mode()[0],
                 'lcxb': training_set['lcxb'].mode()[0],
                 'lcxc': training_set['lcxc'].mode()[0],
                 'ld1': training_set['ld1'].mode()[0],
                 'ld2': training_set['ld2'].mode()[0],
                 'lita': training_set['lita'].mode()[0],
                 'lm': training_set['lm'].mode()[0],
                 'lom1': training_set['lom1'].mode()[0],
                 'lom2': training_set['lom2'].mode()[0],
                 'lpd': training_set['lpd'].mode()[0],
                 'lpl': training_set['lpl'].mode()[0],
                 'ram_rv': training_set['ram_rv'].mode()[0],
                 'rcaa': training_set['rcaa'].mode()[0],
                 'rcab': training_set['rcab'].mode()[0],
                 'rcac': training_set['rcac'].mode()[0],
                 'rita': training_set['rita'].mode()[0],
                 'rpd': training_set['rpd'].mode()[0],
                 'rpl': training_set['rpl'].mode()[0],
                 'vgrca_ag': training_set['vgrca_ag'].mode()[0],
                 'vglca1_ag': training_set['vglca1_ag'].mode()[0],
                 'vglca2_ag': training_set['vglca2_ag'].mode()[0],
                 'restenosis': training_set['restenosis'].mode()[0],
                 'stent_dimension': training_set['stent_dimension'].mean(),
                 'ball_dimension': training_set['ball_dimension'].mean(),
                 'add_stent_1': 0,
                 'add_stent_2_tai_yli': 0}

# combine datasets for imputing

df = training_set.append([validation_set, testing_set])

# impute data

for key, val in impute_values.items():
    df[key] = df[key].fillna(val)

#%% fill in mutually exclusive categorical values

# obtain categorical impute values

sten_post_training = training_set[['sten_post_0', 'sten_post_25', 'sten_post_60', 
                                   'sten_post_85', 'sten_post_100']].idxmax(axis = 1)
impute_values['sten_post'] = sten_post_training.mode()[0]

sten_pre_training = training_set[['sten_pre_100', 'sten_pre_85', 
                                  'sten_pre_60']].idxmax(axis = 1)
impute_values['sten_pre'] = sten_pre_training.mode()[0]

AHA_training = training_set[['AHA_a', 'AHA_b1', 'AHA_b2', 
                             'AHA_c', 'AHA_cto']].idxmax(axis = 1)
impute_values['AHA'] = AHA_training.mode()[0]

# impute data

sten_post = df[['sten_post_0', 'sten_post_25', 'sten_post_60', 
                'sten_post_85', 'sten_post_100']].idxmax(axis = 1)
sten_post = sten_post.fillna(impute_values['sten_post'])
sten_post = pd.get_dummies(sten_post).astype(int)
sten_post = sten_post[['sten_post_0', 'sten_post_25', 'sten_post_60', 
                       'sten_post_85', 'sten_post_100']]
df[['sten_post_0', 'sten_post_25', 'sten_post_60', 'sten_post_85', 
    'sten_post_100']] = sten_post

sten_pre = df[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']].idxmax(axis = 1)
sten_pre = sten_pre.fillna(impute_values['sten_pre'])
sten_pre = pd.get_dummies(sten_pre).astype(int)
sten_pre = sten_pre[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']]
df[['sten_pre_100', 'sten_pre_85', 'sten_pre_60']] = sten_pre

AHA = df[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']].idxmax(axis = 1)
AHA = AHA.fillna(impute_values['AHA'])
AHA = pd.get_dummies(AHA).astype(int)
AHA = AHA[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']]
df[['AHA_a', 'AHA_b1', 'AHA_b2', 'AHA_c', 'AHA_cto']] = AHA

#%% check for nan values

df.isnull().values.any()

#%% split impute data back to training, validation and testing

training_set = df[:n_training]
validation_set = df[n_training:n_training+n_validation]
testing_set = df[-n_testing:]

#%% define feature and target labels

feature_labels = ['paino', 'AHA_cto', 'Patient_sex', 'FN2BA',
                  'I20.81_I21.01_I21.11_or_I21.41', 'add_stent_2_tai_yli',
                  'n_tmp_3', 'sten_post_100', 'suonia_2_tai_yli', 'pituus']

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

training_features = training_set[feature_labels]
validation_features = validation_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
validation_targets = validation_set[target_label]
testing_targets = testing_set[target_label]

#%% scale features

# z-score

t_mean = training_features.mean()
t_std = training_features.std()

training_features = (training_features - t_mean) / t_std
validation_features = (validation_features - t_mean) / t_std
testing_features = (testing_features - t_mean) / t_std

# log (for skewed data)

#training_features = np.log1p(training_features)
#validation_features = np.log1p(validation_features)
#testing_features = np.log1p(testing_features)

# box cox (for skewed data)

#lmbda = 0.15
#
#training_features = sp.special.boxcox1p(training_features, lmbda)
#validation_features = sp.special.boxcox1p(validation_features, lmbda)
#testing_features = sp.special.boxcox1p(testing_features, lmbda)

#%% log transform targets (for skewed data)

training_targets = np.log1p(training_targets)
validation_targets = np.log1p(validation_targets)
testing_targets = np.log1p(testing_targets)

#%% build and train neural network model

# define parameters

learning_rate = 0.001
n_epochs = 150
n_neurons = 64
n_layers = 2
batch_size = 5
l1_reg = 0.0
l2_reg = 0.01
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

#f1 = plot_regression_performance(history, training_targets, training_predictions, 
#                                 validation_targets, validation_predictions)

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
                     'df': df,
                     'df_orig': df_orig,
                     'impute_values': impute_values,
                     'feature_labels': feature_labels,
                     'target_label': target_label,
                     'n_training': n_training,
                     'n_validation': n_validation,
                     'n_testing': n_testing,
                     'training_set': training_set,
                     'training_features': training_features,
                     'training_targets': training_targets,
                     'validation_set': validation_set,
                     'validation_features': validation_features,
                     'validation_targets': validation_targets,
                     'testing_set': testing_set,
                     'testing_features': testing_features,
                     'testing_targets': testing_targets}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')

model.save(model_dir + '\\' + 'keras_model.h5')


