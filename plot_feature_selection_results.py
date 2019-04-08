# -*- coding: utf-8 -*-
'''
Created on Tue Feb 12 14:26:01 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    February 2019
    
@description:
    
    This code is used for plotting feature selection results
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

#%% load variables

dir_name = '20190407-100804_NF40_NM9_NI100'

model_dir = os.path.join('Feature selection', dir_name)

variables = pickle.load(open(os.path.join(model_dir, 'variables.pkl'), 'rb'))

for key,val in variables.items():
        exec(key + '=val')
        
del variables
        
#%% plot figures
        
# define colormap

#cmap = sns.diverging_palette(220, 10, as_cmap = True)
cmap = sns.diverging_palette(250, 10, as_cmap = True)

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

f3 = plt.figure()
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'validation_score', 
                  label = 'Validation', ci = 95)
ax = sns.lineplot(data = reg_summary, x = 'n_features', y = 'test_score', 
                  label = 'Test', ci = 95)
ax.grid(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
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

f5 = plt.figure(figsize = (24, 4))
ax = sns.heatmap(heatmap_rankings_mean, cmap = 'Reds', linewidths = 0.5, annot = True, fmt = '.0f')
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

f6 = plt.figure(figsize = (24, 4))
ax = sns.heatmap(heatmap_rankings_median, cmap = 'Reds', linewidths = 0.5, annot = True, fmt = '.0f')
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
                 square = True, linewidths = 0.5, cbar_kws = {'shrink': 0.5, 'ticks': [-1, 0, 1]})

f10 = plt.figure(figsize = (6, 6))
ax = sns.heatmap(method_corr, mask = method_corr_mask, cmap = cmap, vmin = -1, vmax = 1, center = 0,
                 square = True, linewidths = 0.5, cbar_kws = {'shrink': 0.5, 'ticks': [-1, 0, 1]})

#%% save figures

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
for filetype in ['pdf', 'png', 'eps']:
    
#    f1.savefig(os.path.join(model_dir, ('heatmap_vscore_mean.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f2.savefig(os.path.join(model_dir, ('heatmap_tscore_mean.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f3.savefig(os.path.join(model_dir, ('lineplot_scores.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f4.savefig(os.path.join(model_dir, ('boxplot_feature_rankings.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f5.savefig(os.path.join(model_dir, ('heatmap_rankings_mean.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f6.savefig(os.path.join(model_dir, ('heatmap_rankings_median.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f7.savefig(os.path.join(model_dir, ('parameter_c.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
#    f8.savefig(os.path.join(model_dir, ('parameter_gamma.' + filetype)), dpi = 600, format = filetype,
#               bbox_inches = 'tight', pad_inches = 0)
    f9.savefig(os.path.join(model_dir, ('feature_corr.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f10.savefig(os.path.join(model_dir, ('method_corr.' + filetype)), dpi = 600, format = filetype,
                bbox_inches = 'tight', pad_inches = 0)
