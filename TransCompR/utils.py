# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:54:41 2023

@author: Ran
"""
import pandas as pd
import numpy as np
def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def validate_dataframes(human_data, mouse_data, human_covar):
     # Check if the provided data is pandas DataFrame and not empty
     for df in [human_data, mouse_data, human_covar]:
         if df is not None:
             if not isinstance(df, pd.DataFrame) or df.empty:
                 raise ValueError("Data must be a non-empty pandas DataFrame.")
                 
def validate_hvg_and_pcs(hvg, n_pcs, mouse_data):
     if not isinstance(hvg, int) or hvg < 0 or hvg > mouse_data.shape[1]:
         raise ValueError("If you want to use HVG, please pass a positive integer that is less than the total number of genes in the dataset. Otherwise set hvg to zero.")
         
     if not (isinstance(n_pcs, int) and n_pcs > 0) and n_pcs != 'max':
         raise ValueError("n_pcs must be a positive integer or 'max'.")
 
def validate_classes(human_classes):
     if not isinstance(human_classes, (pd.Series, pd.DataFrame, np.ndarray)):
         raise ValueError("human_classes should be a pandas Series, DataFrame or a numpy array. If human_classes is a list, please use numpy.array(human_classes) instead")
     else:
         if isinstance(human_classes, np.ndarray):
             # convert numpy array to pandas Series
             human_classes = pd.Series(human_classes.flatten(), name='target')
     return human_classes    