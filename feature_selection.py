# -*- coding: utf-8 -*-
'''
Created on Fri Mar  8 10:10:57 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    February 2019
    
@description:
    
    This code is used for feature selection for different regression models
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # only for cluster use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# import feature selection methods

from sklearn.feature_selection import f_regression, mutual_info_regression
#from skfeature.function.information_theoretical_based import CMIM
#from skfeature.function.structure import group_fs, tree_fs
#from skfeature.function.streaming import alpha_investing
#from sklearn_relief import RReliefF
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None                                       # disable imputation warnings

#%% read data

df = pd.read_csv('radiation_dose_data.csv', sep = ',')

#%% check for duplicates

duplicates = any(df.duplicated())

#%% create synthetic features

df['BSA'] = 0.007184 * df['Weight'].pow(0.425) * df['Height'].pow(0.725)
df['BMI'] = df['Weight'] / (df['Height'] / 1e2).pow(2)

#%% drop nan values based on column name(s)

df = df.dropna(subset = ['Weight'])

#%% calculate data statistics

df_stats = pd.DataFrame(df.isnull().mean() * 100, columns = ['NaN ratio'])
df_stats['Mean'] = df.mean()
df_stats['Median'] = df.median()
df_stats['Min'] = df.min()
df_stats['Max'] = df.max()
df_stats['SD'] = df.std()
df_stats['Sum'] = df.sum()

#%% define feature and target labels

feature_labels = ['Weight', 
                  'Height', 
                  'Gender', 
                  'Age', 
                  'I20.81, I21.01, I21.11 or I21.41', 
                  'I35.0', 
                  'FN1AC', 
                  'FN2BA',
                  'FN2AA', 
                  'TFC00', 
                  'N of procedures 1', 
                  'N of procedures 2', 
                  'N of procedures 3', 
                  'PCI in STEMI', 
                  'Flap failure', 
                  'NSTEMI', 
                  'Diagnostic', 
                  'UAP', 
                  'Heart failure', 
                  'STEMI other',
                  'Stable AP', 
                  'Arrhythmia settlement', 
                  'Multi-vessel disease', 
                  'LM unprotected', 
                  'Previous CABG', 
                  'IM', 
                  'LADa', 
                  'LADb',
                  'LADc', 
                  'LCXa', 
                  'LCXb', 
                  'LCXc', 
                  'LD1', 
                  'LD2', 
#                  'Lita',
                  'LM', 
                  'LOM1', 
                  'LOM2', 
                  'LPD', 
                  'LPL', 
#                  'RAM (RV)', 
                  'RCAa', 
                  'RCAb',
                  'RCAc', 
                  'Rita', 
                  'RPD', 
                  'RPL', 
                  'VGRCA (AG)', 
                  'VGLCA1 (AG)', 
#                  'VGLCA2 (AG)', 
                  'Restenosis', 
                  'Stent dimension', 
                  'Ball dimension',
                  'Additional stenting 1', 
                  'Additional stenting over 1', 
                  'Post-stenosis 0%', 
                  'Post-stenosis 25%', 
                  'Post-stenosis 60%', 
                  'Post-stenosis 85%', 
                  'Post-stenosis 100%',
                  'Pre-stenosis 100%', 
                  'Pre-stenosis 85%', 
                  'Pre-stenosis 60%', 
                  'AHA score A', 
                  'AHA score B1',
                  'AHA score B2', 
                  'AHA score C', 
                  'CTO', 
                  'IVUS', 
                  'OCT',
                  'BSA',
                  'BMI'
                  ]

target_label = ['DAP']

#%% define parameters for iteration

# define number of iterations

n_iterations = 10

# define split ratio for training and testing sets

split_ratio = 0.2

# define scaling type ('log', 'minmax', 'standard' or None)

scaling_type = 'log'

# define number of features

n_features = [5, 10, 15, 20, 25, 30, 35, 40]

# define scorer methods

methods =   ['FREG',
             'MIR',
             'PEAR',
             'SPEA',
             'RELF', 
             'SURF', 
             'SURFS', 
             'MSURF', 
             'MSURFS'
             ]

# define scorer functions

scorers = [f_regression,
           mutual_info_regression,
           'pearson',
           'spearman',
           ReliefF,
           SURF,
           SURFstar,
           MultiSURF,
           MultiSURFstar,
           ]

# define parameters for parameter search

grid_param =    {
                'kernel': ['rbf'],
                'epsilon': [0.1],
                'C': list(np.logspace(0, 5, 6)),
                'gamma': list(np.logspace(-1, 3, 5))
                }

# impute features

impute = True

# discretise features

discretise = True

# define regression model

max_iter = 200000

reg_model = SVR(cache_size = 4000, max_iter = max_iter)

# define parameter search method

cv = 5
scoring = 'neg_mean_squared_error'
    
reg_grid = GridSearchCV(reg_model, grid_param, n_jobs = -1, cv = cv, 
                        scoring = scoring, refit = True, iid = False)

# initialise variables

reg_results = pd.DataFrame()
feature_rankings = pd.DataFrame()
k = len(feature_labels)

#%% start the iteration

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

for iteration in range(0, n_iterations):
    
    # define random state

    random_state = np.random.randint(0, 10000)
    
    # assign random state to grid parameters
    
#    grid_param['random_state'] = [random_state]
    
    # print progress
    
    print('Iteration %d with random state %d at %.1f min' % (iteration, random_state, 
                                                             ((time.time() - start_time) / 60)))
    
    # randomise and divive data for cross-validation
    
    training_set, testing_set = train_test_split(df, test_size = split_ratio,
                                                 random_state = random_state)
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # impute features
    
    if impute == True:
    
        impute_mean =   ['Weight', 
                         'Height', 
                         'Stent dimension', 
                         'Ball dimension', 
                         'BSA', 
                         'BMI'
                         ]
        
        impute_mode =   []
        
        impute_cons =   ['PCI in STEMI', 
                         'Flap failure', 
                         'NSTEMI', 
                         'Diagnostic', 
                         'UAP', 
                         'Heart failure', 
                         'STEMI other',
                         'Stable AP', 
                         'Arrhythmia settlement', 
                         'Multi-vessel disease', 
                         'LM unprotected',
                         'IM', 
                         'LADa', 
                         'LADb',
                         'LADc', 
                         'LCXa', 
                         'LCXb', 
                         'LCXc', 
                         'LD1', 
                         'LD2', 
                         #'Lita',
                         'LM', 
                         'LOM1', 
                         'LOM2', 
                         'LPD', 
                         'LPL', 
                         #'RAM (RV)', 
                         'RCAa', 
                         'RCAb',
                         'RCAc', 
                         'Rita', 
                         'RPD', 
                         'RPL', 
                         'VGRCA (AG)', 
                         'VGLCA1 (AG)', 
                         #'VGLCA2 (AG)', 
                         'Restenosis',
                         'Additional stenting 1', 
                         'Additional stenting over 1',
                         'Post-stenosis 0%', 
                         'Post-stenosis 25%', 
                         'Post-stenosis 60%', 
                         'Post-stenosis 85%', 
                         'Post-stenosis 100%',
                         'Pre-stenosis 100%', 
                         'Pre-stenosis 85%', 
                         'Pre-stenosis 60%', 
                         'AHA score A', 
                         'AHA score B1',
                         'AHA score B2', 
                         'AHA score C', 
                         'CTO'
                         ]
        
        imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imp_cons = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
        
        training_features[impute_mean] = imp_mean.fit_transform(training_features[impute_mean])
        testing_features[impute_mean] = imp_mean.transform(testing_features[impute_mean])
        
        training_features[impute_mode] = imp_mode.fit_transform(training_features[impute_mode])
        testing_features[impute_mode] = imp_mode.transform(testing_features[impute_mode])
        
        training_features[impute_cons] = imp_cons.fit_transform(training_features[impute_cons])
        testing_features[impute_cons] = imp_cons.transform(testing_features[impute_cons])
        
        del imp_mean, imp_mode, imp_cons
    
    # discretise features
    
    if discretise == True:
    
        disc_labels =   ['Weight', 
                         'Height', 
                         'Age',
                         'Stent dimension', 
                         'Ball dimension', 
                         'BSA', 
                         'BMI'
                         ]
        
        enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
        
        training_features[disc_labels] = enc.fit_transform(training_features[disc_labels])
        testing_features[disc_labels] = enc.transform(testing_features[disc_labels])
        
        del enc
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        scaler = MinMaxScaler(feature_range = (0, 1)) 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
        
        del scaler
        
    elif scaling_type == 'standard':
        
        scaler = StandardScaler() 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
        
        del scaler
    
    # find k best features for each feature selection method
    
    k_features = pd.DataFrame(index = range(0, k), columns = methods)
    
    for scorer, method in zip(scorers, methods):
            
        if method in ('FREG'):
            
            scores, _ = scorer(training_features.values, training_targets.values[:, 0])
            indices = np.argsort(scores)[::-1]
            k_features[method] = list(training_features.columns.values[indices[0:k]])
            
            del scores, indices
            
        elif method in ('MIR'):
            
            scores = scorer(training_features.values, training_targets.values[:, 0])
            indices = np.argsort(scores)[::-1]
            k_features[method] = list(training_features.columns.values[indices[0:k]])
            
            del scores, indices
            
        elif method in ('PEAR', 'SPEA'):
            
            fcorr = pd.concat([training_features, training_targets], axis = 1, sort = False).corr(method = scorer)
            scores = abs(fcorr[target_label].drop(target_label, axis = 0))
            k_features[method] = list(scores.sort_values(by = target_label, ascending = False).index)
            
            del fcorr, scores
            
        elif method in ('RELF', 'SURF', 'SURFS', 'MSURF', 'MSURFS'):
            
            rebate = scorer(n_features_to_select = k, n_jobs = -1)
            reg = rebate.fit(training_features.values, training_targets.values[:, 0])
            indices = np.argsort(reg.feature_importances_)[::-1]
            k_features[method] = list(training_features.columns.values[indices[0:k]])
            
            del rebate, reg, indices   
            
    del scorer, method
    
    # calculate feature scores
    
    k_rankings = pd.DataFrame(k_features.T.values.argsort(1),
                              columns = np.sort(k_features.iloc[:, 0].values),
                              index = k_features.columns)
    k_rankings['method'] = k_rankings.index
    k_rankings['iteration'] = iteration
    k_rankings['random_state'] = random_state
    feature_rankings = feature_rankings.append(k_rankings, sort = False, ignore_index = True)
    
    del k_rankings
    
    # train model using parameter search

    for n in n_features:
        for method in methods:
            
            # fit parameter search
        
            reg_fit = reg_grid.fit(training_features[k_features[method][0:n]].values, training_targets.values[:, 0])
            
            # calculate predictions
            
            testing_predictions = reg_fit.predict(testing_features[k_features[method][0:n]].values)
            test_score = mean_squared_error(testing_targets.values[:, 0], testing_predictions)
            
            # save results
            
            res = pd.DataFrame(reg_fit.best_params_, index = [0])
            res['method'] = method
            res['validation_score'] = abs(reg_fit.best_score_)
            res['test_score'] = test_score
            res['n_features'] = n
            res['iteration'] = iteration
            res['random_state'] = random_state
            reg_results = reg_results.append(res, sort = False, ignore_index = True)
            
            del reg_fit, testing_predictions, test_score, res
    
    del n, method
    del k_features, random_state
    del training_set, training_features, training_targets
    del testing_set, testing_features, testing_targets
    
del iteration

end_time = time.time()

print('Total execution time: %.1f min' % ((end_time - start_time) / 60))

#%% calculate summaries

# summarise results

mean_vscores = reg_results.groupby(['method', 'n_features'], as_index = False)['validation_score'].mean()
mean_tscores = reg_results.groupby(['method', 'n_features'])['test_score'].mean().values

std_vscores = reg_results.groupby(['method', 'n_features'])['validation_score'].std().values
std_tscores = reg_results.groupby(['method', 'n_features'])['test_score'].std().values

reg_summary = mean_vscores.copy()
reg_summary['test_score'] = mean_tscores
reg_summary['validation_score_std'] =  std_vscores
reg_summary['test_score_std'] = std_tscores

del mean_vscores, mean_tscores, std_vscores, std_tscores

# calculate heatmaps for test scores, validation scores and feature reankings
    
heatmap_vscore_mean = reg_summary.pivot(index = 'method', columns = 'n_features', values = 'validation_score')
heatmap_vscore_mean.columns = heatmap_vscore_mean.columns.astype(int)

heatmap_tscore_mean = reg_summary.pivot(index = 'method', columns = 'n_features', values = 'test_score')
heatmap_tscore_mean.columns = heatmap_tscore_mean.columns.astype(int)

heatmap_rankings_mean = feature_rankings.groupby(['method'], as_index = False)[feature_labels].mean()
heatmap_rankings_mean = heatmap_rankings_mean.set_index('method')

heatmap_rankings_median = feature_rankings.groupby(['method'], as_index = False)[feature_labels].median()
heatmap_rankings_median = heatmap_rankings_median.set_index('method')

# calculate box plot

feature_boxplot = feature_rankings[feature_labels].melt(var_name = 'feature', value_name = 'ranking')

# calculate top features based on mean and median values

top_features_mean = feature_boxplot.groupby(['feature'], as_index = False)['ranking'].mean()
top_features_mean['std'] = feature_boxplot.groupby(['feature'])['ranking'].std().values
top_features_mean = top_features_mean.sort_values('ranking', ascending = True)
top_features_mean = top_features_mean.reset_index(drop = True)
top_features_mean['method'] = 'TOPN'

top_features_median = feature_boxplot.groupby(['feature'], as_index = False)['ranking'].median()
top_features_median['std'] = feature_boxplot.groupby(['feature'])['ranking'].std().values
top_features_median = top_features_median.sort_values('ranking', ascending = True)
top_features_median = top_features_median.reset_index(drop = True)
top_features_median['method'] = 'TOPN'

#%% train model with only top features

top_results = pd.DataFrame()
random_states = reg_results.groupby(['iteration'])['random_state'].mean().values
iteration = 0

time_stamp = time.time()

for random_state in random_states:
    
    # assign random state to grid parameters
    
#    grid_param['random_state'] = [random_state]
    
    # print progress
    
    print('Iteration %d with random state %d at %.1f min' % (iteration, random_state, 
                                                             ((time.time() - time_stamp) / 60)))
    
    # randomise and divive data for cross-validation
    
    training_set, testing_set = train_test_split(df, test_size = split_ratio,
                                                 random_state = random_state)
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # impute features
    
    if impute == True:
    
        impute_mean =   ['Weight', 
                         'Height', 
                         'Stent dimension', 
                         'Ball dimension', 
                         'BSA', 
                         'BMI'
                         ]
        
        impute_mode =   ['PCI in STEMI', 
                         'Flap failure', 
                         'NSTEMI', 
                         'Diagnostic', 
                         'UAP', 
                         'Heart failure', 
                         'STEMI other',
                         'Stable AP', 
                         'Arrhythmia settlement', 
                         'Multi-vessel disease', 
                         'LM unprotected',
                         'IM', 
                         'LADa', 
                         'LADb',
                         'LADc', 
                         'LCXa', 
                         'LCXb', 
                         'LCXc', 
                         'LD1', 
                         'LD2', 
                         #'Lita',
                         'LM', 
                         'LOM1', 
                         'LOM2', 
                         'LPD', 
                         'LPL', 
                         #'RAM (RV)', 
                         'RCAa', 
                         'RCAb',
                         'RCAc', 
                         'Rita', 
                         'RPD', 
                         'RPL', 
                         'VGRCA (AG)', 
                         'VGLCA1 (AG)', 
                         #'VGLCA2 (AG)', 
                         'Restenosis', 
                         'Post-stenosis 0%', 
                         'Post-stenosis 25%', 
                         'Post-stenosis 60%', 
                         'Post-stenosis 85%', 
                         'Post-stenosis 100%',
                         'Pre-stenosis 100%', 
                         'Pre-stenosis 85%', 
                         'Pre-stenosis 60%', 
                         'AHA score A', 
                         'AHA score B1',
                         'AHA score B2', 
                         'AHA score C', 
                         'CTO'
                         ]
        
        impute_cons =   ['Additional stenting 1', 
                         'Additional stenting over 1'
                         ]
        
        imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imp_cons = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
        
        training_features[impute_mean] = imp_mean.fit_transform(training_features[impute_mean])
        testing_features[impute_mean] = imp_mean.transform(testing_features[impute_mean])
        
        training_features[impute_mode] = imp_mode.fit_transform(training_features[impute_mode])
        testing_features[impute_mode] = imp_mode.transform(testing_features[impute_mode])
        
        training_features[impute_cons] = imp_cons.fit_transform(training_features[impute_cons])
        testing_features[impute_cons] = imp_cons.transform(testing_features[impute_cons])
        
        del imp_mean, imp_mode, imp_cons
    
    # discretise features
    
    if discretise == True:
    
        disc_labels =   ['Weight', 
                         'Height', 
                         'Age',
                         'Stent dimension', 
                         'Ball dimension', 
                         'BSA', 
                         'BMI'
                         ]
        
        enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
        
        training_features[disc_labels] = enc.fit_transform(training_features[disc_labels])
        testing_features[disc_labels] = enc.transform(testing_features[disc_labels])
        
        del enc
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        scaler = MinMaxScaler(feature_range = (0, 1)) 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
        
    elif scaling_type == 'standard':
        
        scaler = StandardScaler() 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
    
    for n in n_features:
        
        # fit parameter search
            
        reg_fit = reg_grid.fit(training_features[top_features_median['feature'][0:n]].values, training_targets.values[:, 0])
        
        # calculate predictions
        
        testing_predictions = reg_fit.predict(testing_features[top_features_median['feature'][0:n]].values)
        test_score = mean_squared_error(testing_targets.values[:, 0], testing_predictions)
        
        # save results
        
        res = pd.DataFrame(reg_fit.best_params_, index = [0])
        res['method'] = 'TOPN'
        res['validation_score'] = abs(reg_fit.best_score_)
        res['test_score'] = test_score
        res['n_features'] = n
        res['iteration'] = iteration
        res['random_state'] = random_state
        top_results = top_results.append(res, sort = False, ignore_index = True)
        
        del reg_fit, testing_predictions, test_score, res
        
    del n
    del training_set, training_features, training_targets
    del testing_set, testing_features, testing_targets
        
    iteration += 1

print('Total execution time: %.1f min' % ((time.time() - time_stamp) / 60))

del random_state, iteration, time_stamp

#%% calculate top summaries

# summarise results

mean_vscores = top_results.groupby(['method', 'n_features'], as_index = False)['validation_score'].mean()
mean_tscores = top_results.groupby(['method', 'n_features'])['test_score'].mean().values

std_vscores = top_results.groupby(['method', 'n_features'])['validation_score'].std().values
std_tscores = top_results.groupby(['method', 'n_features'])['test_score'].std().values

top_summary = mean_vscores.copy()
top_summary['test_score'] = mean_tscores
top_summary['validation_score_std'] =  std_vscores
top_summary['test_score_std'] = std_tscores

del mean_vscores, mean_tscores, std_vscores, std_tscores

# calculate heatmaps for test scores, validation scores and feature reankings
    
top_vscore_mean = top_summary.pivot(index = 'method', columns = 'n_features', values = 'validation_score')
top_vscore_mean.columns = top_vscore_mean.columns.astype(int)

top_tscore_mean = top_summary.pivot(index = 'method', columns = 'n_features', values = 'test_score')
top_tscore_mean.columns = top_tscore_mean.columns.astype(int)

top_rankings_mean = top_features_mean.pivot(index = 'method', columns = 'feature', values = 'ranking')
top_rankings_median = top_features_median.pivot(index = 'method', columns = 'feature', values = 'ranking')

# append top scores into existing heatmaps

heatmap_vscore_mean = heatmap_vscore_mean.append(top_vscore_mean, sort = True, ignore_index = False)
heatmap_tscore_mean = heatmap_tscore_mean.append(top_tscore_mean, sort = True, ignore_index = False)

heatmap_rankings_mean = heatmap_rankings_mean.append(top_rankings_mean, sort = True, ignore_index = False)
heatmap_rankings_median = heatmap_rankings_median.append(top_rankings_median, sort = True, ignore_index = False)

del top_vscore_mean, top_tscore_mean, top_rankings_mean, top_rankings_median

#%% calculate feature correlations

# correlation matrix

feature_corr = df[feature_labels].corr(method = 'spearman')
method_corr = heatmap_rankings_median.T.corr(method = 'kendall')

# a mask for the upper triangle

feature_corr_mask = np.zeros_like(feature_corr, dtype = np.bool)
feature_corr_mask[np.triu_indices_from(feature_corr_mask)] = True

method_corr_mask = np.zeros_like(method_corr, dtype = np.bool)
method_corr_mask[np.triu_indices_from(method_corr_mask)] = True

#%% plot figures

# define colormap

#cmap = sns.diverging_palette(220, 10, as_cmap = True)
#cmap = sns.diverging_palette(250, 10, as_cmap = True)
cmap = 'RdGy'

# plot validation and test scores

f1 = plt.figure(figsize = (8, 4))
ax = sns.heatmap(heatmap_vscore_mean, cmap = 'Reds', linewidths = 0.5, annot = True, fmt = ".0f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f2 = plt.figure(figsize = (8, 4))
ax = sns.heatmap(heatmap_tscore_mean, cmap = 'Reds', linewidths = 0.5, annot = True, fmt = ".0f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f3 = plt.figure(figsize = (6, 4))
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'validation_score', 
                  label = 'Validation', ci = 95, color = 'orangered')
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'test_score', 
                  label = 'Test', ci = 95, color = 'k')
ax.grid(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.autoscale(enable = True, axis = 'x', tight = True)
plt.legend(loc = 'upper right')
plt.ylabel('Mean error')
plt.xlabel('Number of features')

# plot feature rankings

f4 = plt.figure(figsize = (16, 4))
ax = sns.boxplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = top_features_median['feature'],
                 whis = 1.5, palette = 'Reds', fliersize = 2, notch = True)
#ax = sns.swarmplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = feature_order, 
#                   size = 2, color = '.3', linewidth = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.ylabel('Ranking')
plt.xlabel('Feature')

f5 = plt.figure(figsize = (22, 4))
ax = sns.heatmap(heatmap_rankings_mean, cmap = 'Reds', linewidths = 0.5, annot = True, 
                 fmt = '.0f', cbar_kws = {'pad': 0.01})
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

f6 = plt.figure(figsize = (22, 4))
ax = sns.heatmap(heatmap_rankings_median, cmap = 'Reds', linewidths = 0.5, annot = True, 
                 fmt = '.0f', cbar_kws = {'pad': 0.01})
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

# plot parameter distributions

f7 = plt.figure(figsize = (6, 4))
ax = reg_results.C.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('C')

f8 = plt.figure(figsize = (6, 4))
ax = reg_results.gamma.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('Gamma')

# plot correlations

f9 = plt.figure(figsize = (16, 16))
ax = sns.heatmap(feature_corr, mask = feature_corr_mask, cmap = cmap, vmin = -1, vmax = 1, center = 0,
                 square = True, linewidths = 0.5, cbar_kws = {'shrink': 0.3, 'ticks': [-1, 0, 1],
                                                              'pad': -0.05})

f10 = plt.figure(figsize = (6, 6))
ax = sns.heatmap(method_corr, mask = method_corr_mask, cmap = cmap, vmin = -1, vmax = 1, center = 0,
                 square = True, linewidths = 0.5, cbar_kws = {'shrink': 0.5, 'ticks': [-1, 0, 1],
                                                              'pad': -0.05})

#%% save data

# make directory

model_dir = os.path.join('Feature selection', 
                         ('%s_NF%d_NM%d_NI%d' % (timestr, max(n_features), len(methods), n_iterations)))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# save parameters into text file
    
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as text_file:
    text_file.write('timestr: %s\n' % timestr)
    text_file.write('Computation time: %.1f min\n' % ((end_time - start_time) / 60))
    text_file.write('Number of samples: %d\n' % len(df))
    text_file.write('Number of features: %d\n' % len(feature_labels))
    text_file.write('methods: %s\n' % str(methods))
    text_file.write('duplicates: %s\n' % str(duplicates))
    text_file.write('n_iterations: %d\n' % n_iterations)
    text_file.write('discretise: %s\n' % str(discretise))
    text_file.write('impute: %s\n' % str(impute))
    text_file.write('scaling_type: %s\n' % scaling_type)
    text_file.write('scoring: %s\n' % scoring)
    text_file.write('split_ratio: %.1f\n' % split_ratio)
    text_file.write('cv: %d\n' % cv)
    
# save figures
    
for filetype in ['pdf', 'png', 'eps']:
    
    f1.savefig(os.path.join(model_dir, ('heatmap_vscore_mean.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f2.savefig(os.path.join(model_dir, ('heatmap_tscore_mean.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f3.savefig(os.path.join(model_dir, ('lineplot_scores.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f4.savefig(os.path.join(model_dir, ('boxplot_feature_rankings.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f5.savefig(os.path.join(model_dir, ('heatmap_rankings_mean.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f6.savefig(os.path.join(model_dir, ('heatmap_rankings_median.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f7.savefig(os.path.join(model_dir, ('parameter_c.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f8.savefig(os.path.join(model_dir, ('parameter_gamma.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f9.savefig(os.path.join(model_dir, ('feature_corr.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f10.savefig(os.path.join(model_dir, ('method_corr.' + filetype)), dpi = 600, format = filetype,
                bbox_inches = 'tight', pad_inches = 0)

# save variables
    
variable_names = %who_ls DataFrame ndarray list dict str bool int int64 float float64
variables = dict((name, eval(name)) for name in variable_names)
    
pickle.dump(variables, open(os.path.join(model_dir, 'variables.pkl'), 'wb'))
