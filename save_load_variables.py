# -*- coding: utf-8 -*-
'''
Created on Fri Sep  7 09:30:00 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This function is used to save variables from workspace into a pickle file
    
'''

#%% import necessary packages

import pickle
import os

#%% save or load variables

def save_load_variables(directory, variables, fname, opt):
    
    '''
    Args:
        directory: path to file
        variables: variables to save/load/add
        fname: filename
        opt: whether save ('save'), load ('load') or add ('add') variables
        
    Returns:
        variables: loaded variables
    '''
    
    file_path = os.path.join(directory, (fname + '.pickle'))
    
    if opt == 'save':
        
        pickle_out = open(file_path, 'wb')
        pickle.dump(variables, pickle_out)
        pickle_out.close()
        
    elif opt == 'load':
        
        pickle_in = open(file_path, 'rb')
        variables = pickle.load(pickle_in)
        pickle_in.close()
        return variables
    
    elif opt == 'add':
        
        pickle_in = open(file_path, 'rb')
        old_variables = pickle.load(pickle_in)
        pickle_in.close()
        new_variables = {**old_variables, **variables}
        pickle_out = open(file_path, 'wb')
        pickle.dump(new_variables, pickle_out)
        pickle_out.close()
        
    else:
        
        print('Invalid option')