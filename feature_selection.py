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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import f_regression, mutual_info_regression
#from skfeature.function.information_theoretical_based import CMIM
#from skfeature.function.structure import group_fs, tree_fs
#from skfeature.function.streaming import alpha_investing

from save_load_variables import save_load_variables

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None                                       # disable imputation warnings

#%% read data

df = pd.read_csv(r'radiation_dose_data.csv', sep = ',')

#%% check for duplicates

duplicates = any(df.duplicated())

#%% create synthetic features

df['BSA'] = 0.007184 * df['paino'].pow(0.425) * df['pituus'].pow(0.725)

#%% calculate nan percent for each label

nan_percent = pd.DataFrame(df.isnull().mean() * 100, columns = ['NaN ratio'])

#%% calculate standard deviation

std = pd.DataFrame(df.std(), columns = ['STD'])

#%% define feature and target labels

feature_labels = ['BSA',
                  'paino', 
                  'pituus', 
                  'Patient_sex', 
                  'Age', 
                  'I20.81_I21.01_I21.11_or_I21.41', 
                  'I35.0', 
                  'FN1AC', 
                  'FN2BA',
                  'FN2AA', 
                  'TFC00', 
                  'n_tmp_1', 
                  'n_tmp_2', 
                  'n_tmp_3', 
                  'ind_pci_in_stemi', 
                  'ind_flap_failure', 
                  'ind_nstemi', 
                  'ind_diag', 
                  'ind_uap', 
                  'ind_heart_failure', 
                  'ind_stemi_other',
                  'ind_stable_ap', 
                  'ind_arrhythmia_settl', 
                  'suonia_2_tai_yli', 
                  'lm_unprotected', 
                  'Aiempi_ohitusleikkaus', 
                  'im', 
                  'lada', 
                  'ladb',
                  'ladc', 
                  'lcxa', 
                  'lcxb', 
                  'lcxc', 
                  'ld1', 
                  'ld2', 
#                  'lita',
                  'lm', 
                  'lom1', 
                  'lom2', 
                  'lpd', 
                  'lpl', 
#                  'ram_rv', 
                  'rcaa', 
                  'rcab',
                  'rcac', 
                  'rita', 
                  'rpd', 
                  'rpl', 
                  'vgrca_ag', 
                  'vglca1_ag', 
#                  'vglca2_ag', 
                  'restenosis', 
                  'stent_dimension', 
                  'ball_dimension',
                  'add_stent_1', 
                  'add_stent_2_tai_yli', 
                  'sten_post_0', 
                  'sten_post_25', 
                  'sten_post_60', 
                  'sten_post_85', 
                  'sten_post_100',
                  'sten_pre_100', 
                  'sten_pre_85', 
                  'sten_pre_60', 
                  'AHA_a', 
                  'AHA_b1',
                  'AHA_b2', 
                  'AHA_c', 
                  'AHA_cto', 
                  'IVUS', 
                  'OCT'
                  ]

target_label = ['Korjattu_DAP_GYcm2']

#%% define parameters for iteration

# define number of iterations

n_iterations = 10

# define split ratio for training and testing sets

split_ratio = 0.2

# define scaling type ('log', 'minmax', 'standard' or None)

scaling_type = 'log'

# define number of features

n_features = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# define scorer methods

methods =   ['FREG',
             'MIR'
             ]

# define scorer functions

scorers = [f_regression,
           mutual_info_regression
           ]

# define parameters for parameter search

grid_param =    {
                'kernel': ['rbf'], 
                'C': list(np.logspace(-1, 4, 6)),
                'gamma': list(np.logspace(-2, 4, 7))
                }

# define data imputation values

impute_labels = ['BSA',
                 'paino', 
                 'pituus',
                 'ind_pci_in_stemi', 
                 'ind_flap_failure', 
                 'ind_nstemi', 
                 'ind_diag', 
                 'ind_uap', 
                 'ind_heart_failure', 
                 'ind_stemi_other',
                 'ind_stable_ap', 
                 'ind_arrhythmia_settl', 
                 'suonia_2_tai_yli', 
                 'lm_unprotected',
                 'im', 
                 'lada', 
                 'ladb',
                 'ladc', 
                 'lcxa', 
                 'lcxb', 
                 'lcxc', 
                 'ld1', 
                 'ld2', 
                 'lita',
                 'lm', 
                 'lom1', 
                 'lom2', 
                 'lpd', 
                 'lpl', 
                 'ram_rv', 
                 'rcaa', 
                 'rcab',
                 'rcac', 
                 'rita', 
                 'rpd', 
                 'rpl', 
                 'vgrca_ag', 
                 'vglca1_ag', 
                 'vglca2_ag', 
                 'restenosis', 
                 'stent_dimension', 
                 'ball_dimension',
                 'add_stent_1', 
                 'add_stent_2_tai_yli', 
                 'sten_post_0', 
                 'sten_post_25', 
                 'sten_post_60', 
                 'sten_post_85', 
                 'sten_post_100',
                 'sten_pre_100', 
                 'sten_pre_85', 
                 'sten_pre_60', 
                 'AHA_a', 
                 'AHA_b1',
                 'AHA_b2', 
                 'AHA_c', 
                 'AHA_cto',
                 ]

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
    
    impute_values = {}
    
    for label in impute_labels:
        
        if label in {'BSA', 'paino', 'pituus', 'Age', 'stent_dimension', 'ball_dimension'}:
            
            impute_values[label] = training_set[label].mean()
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
        else:
            
            impute_values[label] = training_set[label].mode()[0]
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
    del label
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        scaler = MinMaxScaler(feature_range = (0, 1)) 
        training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                         columns = training_features.columns,
                                         index = training_features.index)
        testing_features = pd.DataFrame(scaler.transform(testing_features),
                                        columns = testing_features.columns,
                                        index = testing_features.index)
        
    elif scaling_type == 'standard':
        
        scaler = StandardScaler() 
        training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                         columns = training_features.columns,
                                         index = training_features.index)
        testing_features = pd.DataFrame(scaler.transform(testing_features),
                                        columns = testing_features.columns,
                                        index = testing_features.index)
    
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
            res['validation_score'] = reg_fit.best_score_
            res['test_score'] = test_score
            res['n_features'] = n
            res['iteration'] = iteration
            res['random_state'] = random_state
            reg_results = reg_results.append(res, sort = False, ignore_index = True)
            
            del reg_fit, testing_predictions, test_score, res
    
    del n, method
    del k_features, random_state, impute_values
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
random_states = reg_results.random_state.unique()
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
    
    impute_values = {}
    
    for label in impute_labels:
        
        if label in {'BSA', 'paino', 'pituus', 'Age', 'stent_dimension', 'ball_dimension'}:
            
            impute_values[label] = training_set[label].mean()
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
        else:
            
            impute_values[label] = training_set[label].mode()[0]
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
    del label
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        scaler = MinMaxScaler(feature_range = (0, 1)) 
        training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                         columns = training_features.columns,
                                         index = training_features.index)
        testing_features = pd.DataFrame(scaler.transform(testing_features),
                                        columns = testing_features.columns,
                                        index = testing_features.index)
        
    elif scaling_type == 'standard':
        
        scaler = StandardScaler() 
        training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                         columns = training_features.columns,
                                         index = training_features.index)
        testing_features = pd.DataFrame(scaler.transform(testing_features),
                                        columns = testing_features.columns,
                                        index = testing_features.index)
    
    for n in n_features:
        
        # fit parameter search
            
        reg_fit = reg_grid.fit(training_features[top_features_median['feature'][0:n]].values, training_targets.values[:, 0])
        
        # calculate predictions
        
        testing_predictions = reg_fit.predict(testing_features[top_features_median['feature'][0:n]].values)
        test_score = mean_squared_error(testing_targets.values[:, 0], testing_predictions)
        
        # save results
        
        res = pd.DataFrame(reg_fit.best_params_, index = [0])
        res['method'] = 'TOPN'
        res['validation_score'] = reg_fit.best_score_
        res['test_score'] = test_score
        res['n_features'] = n
        res['iteration'] = iteration
        res['random_state'] = random_state
        top_results = top_results.append(res, sort = False, ignore_index = True)
        
        del reg_fit, testing_predictions, test_score, res
        
    del n
    del impute_values
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

feature_corr = df[feature_labels].corr()

# a mask for the upper triangle

corr_mask = np.zeros_like(feature_corr, dtype = np.bool)
corr_mask[np.triu_indices_from(corr_mask)] = True

#%% plot figures

# define colormap

cmap = sns.diverging_palette(220, 10, as_cmap = True)

# plot validation and test scores

f1 = plt.figure()
ax = sns.heatmap(heatmap_vscore_mean, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".2f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f2 = plt.figure()
ax = sns.heatmap(heatmap_tscore_mean, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".2f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f3 = plt.figure()
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'validation_score', 
                  label = 'Validation', ci = 95)
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'test_score', 
                  label = 'Test', ci = 95)
ax.grid(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.autoscale(enable = True, axis = 'x', tight = True)
plt.legend(loc = 'lower right')
plt.ylabel('Mean score')
plt.xlabel('Number of features')

# plot feature rankings

f4 = plt.figure(figsize = (16, 4))
ax = sns.boxplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = top_features_median['feature'],
                 whis = 1.5, palette = 'Blues', fliersize = 2, notch = True)
#ax = sns.swarmplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = feature_order, 
#                   size = 2, color = '.3', linewidth = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.ylabel('Ranking')
plt.xlabel('Feature')

f5 = plt.figure(figsize = (22, 4))
ax = sns.heatmap(heatmap_rankings_mean, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = '.1f')
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

f6 = plt.figure(figsize = (18, 4))
ax = sns.heatmap(heatmap_rankings_median, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = '.0f')
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

# plot parameter distributions

f7 = plt.figure()
ax = reg_results.C.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('C')

f8 = plt.figure()
ax = reg_results.gamma.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('Gamma')

# plot feature correlations

f9 = plt.figure(figsize = (16, 16))
ax = sns.heatmap(feature_corr, mask = corr_mask, cmap = cmap, vmin = -1, vmax = 1, center = 0,
            square = True, linewidths = 0.5, cbar_kws = {'shrink': 0.5, 'ticks': [-1, 0, 1]})

#%% save figures and variables

model_dir = 'Feature selection\\%s_NF%d_NM%d_NI%d' % (timestr, max(n_features), len(methods), n_iterations)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
for filetype in ['pdf', 'png', 'eps']:
    
    f1.savefig(model_dir + '\\' + 'heatmap_vscore_mean.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f2.savefig(model_dir + '\\' + 'heatmap_tscore_mean.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f3.savefig(model_dir + '\\' + 'lineplot_scores.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f4.savefig(model_dir + '\\' + 'boxplot_feature_rankings.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f5.savefig(model_dir + '\\' + 'heatmap_rankings_mean.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f6.savefig(model_dir + '\\' + 'heatmap_rankings_median.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f7.savefig(model_dir + '\\' + 'parameter_c.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f8.savefig(model_dir + '\\' + 'parameter_gamma.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f9.savefig(model_dir + '\\' + 'feature_corr.' + filetype, dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'nan_percent': nan_percent,
                     'grid_param': grid_param,
                     'impute_labels': impute_labels,
                     'max_iter': max_iter,
                     'k': k,
                     'cv': cv,
                     'scoring': scoring,
                     'n_features': n_features,
                     'n_iterations': n_iterations,
                     'methods': methods,
                     'reg_results': reg_results,
                     'reg_summary': reg_summary,
                     'top_results': top_results,
                     'top_summary': top_summary,
                     'feature_corr': feature_corr,
                     'corr_mask': corr_mask,
                     'feature_rankings': feature_rankings,
                     'feature_boxplot': feature_boxplot,
                     'top_features_mean': top_features_mean,
                     'top_features_median': top_features_median,
                     'heatmap_rankings_mean': heatmap_rankings_mean,
                     'heatmap_rankings_median': heatmap_rankings_median,
                     'heatmap_vscore_mean': heatmap_vscore_mean,
                     'heatmap_tscore_mean': heatmap_tscore_mean,
                     'start_time': start_time,
                     'end_time': end_time,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'scaling_type': scaling_type,
                     'model_dir': model_dir,
                     'df': df,
                     'feature_labels': feature_labels,
                     'target_label': target_label}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')
