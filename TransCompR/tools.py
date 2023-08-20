# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:51:02 2023

@author: Ran
"""

import time
from Bio import Entrez
import pandas as pd
import numpy as np
import warnings
import gseapy as gp
from .utils import  indices, validate_dataframes, validate_hvf_and_pcs, validate_classes

def gsea(results, metric='db', species='Human', top_n_pcs=5, pcs=None, top_n_features=100, save_path=None, **kwargs):
    
    """
    Run GSEA on the top genes of the principal components (PCs) that best differentiate two organism2 classes.
    
    Parameters
    ----------
    results : dict
        A dictionary containing results from previous computations. Should contain keys 
        'predictivity_summary' and 'organism2_transComps'.
    
    metric : str, optional
        Metric for selecting top principal components. Valid options include 'db', 'coefs',
        'indiv acc', 'ch', 'mean diff', and 'tp'. Default is 'db'.
    
    species : str
        Specifies the species for which the GSEA should be performed. Must be one of 
        'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', or 'Worm'.
    
    top_n_pcs : int, optional
        Consider the top N principal components. Default is 5.
    
    pcs : int or list of int, optional
        Specific PCs to be used. When specified, these PCs are used in place of top PCs.
    
    top_n_features : int
        Top N features on both sides of each PC for GSEA.
    
    save_path : str, optional
        Directory where the GSEA table CSV file should be saved. If None, GSEA results 
        won't be saved locally. Default is None.
    
    kwargs : dict, optional
        Additional keyword arguments to be passed to the function.
    
    Returns
    -------
    GSEA_up_result, GSEA_down_result : DataFrame
        The GSEA result tables for upregulated and downregulated genes, respectively.
    
    Notes
    -----
    The function analyzes the principal components (PCs) that best differentiate two 
    classes of organism2 based on a specified metric.
    """
    warnings.filterwarnings("ignore")
    if species not in ['Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm']:
        raise ValueError("Invalid species. Please choose from ['Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'].")

    if metric == 'db':
        sorted_df = results['predictivity_summary'].sort_values(by='davies_bouldin_score', ascending=True)
    elif metric == 'coefs':
        if sum(results['predictivity_summary']['coefs'] != 0) >= top_n_pcs:
            results['predictivity_summary']['coefs_abs'] = abs(results['predictivity_summary']['coefs'])
            sorted_df = results['predictivity_summary'].sort_values(by='coefs_abs', ascending=False)
        else:
            raise ValueError("You don't have enough pcs that have a non-zero coef. Decrease your top_n_pcs.")
    elif metric == 'indiv acc':
        sorted_df = results['predictivity_summary'].sort_values(by='individual_predictivity', ascending=False)
    elif metric == 'ch':
        sorted_df = results['predictivity_summary'].sort_values(by='Calinski-Harabasz Score', ascending=False)
    elif metric == 'mean diff':
        sorted_df = results['predictivity_summary'].sort_values(by='Difference of Means', ascending=False)
    elif metric == 'tp':
        sorted_df = results['predictivity_summary'].sort_values(by='t-test p-value', ascending=True)
    else:
        raise ValueError("Invalid metric. Please choose one from 'coefs', 'indiv acc', 'db','ch','mean diff','tp'.")

    top_ids = sorted_df.index[0:top_n_pcs]
    if pcs is not None:
        top_ids = list(pcs)

    # Fetching the features of organism1 based on sorting, then retrieving the corresponding features from organism2
    top_pc_features_org1 = sorted_df['sorted_organism1_features'].loc[top_ids]
    organism1_features = results['organism1_features']
    organism2_features = results['organism2_features']

    # Retrieve the corresponding organism2 features based on the indices of the features in organism1
    up_features_org2 = []
    down_features_org2 = []
    for features in top_pc_features_org1:
        indices_up = [organism1_features.get_loc(feature) for feature in features[-top_n_features:]]
        indices_down = [organism1_features.get_loc(feature) for feature in features[:top_n_features]]
        up_features_org2.append([organism2_features[idx] for idx in indices_up])
        down_features_org2.append([organism2_features[idx] for idx in indices_down])

    GSEA_up_result = {}
    for i in range(0, len(top_ids)):
        GOBP_up = gp.enrichr(gene_list=up_features_org2[i],
                             gene_sets=['GO_Biological_Process_2021'],
                             organism=species)
        GSEA_up_result[int(top_ids[i])] = GOBP_up.results
        
        GOBP_up.results['-log10_p_adj'] = -np.log10(GOBP_up.results['Adjusted P-value'])
        GOBP_up.results.index = GOBP_up.results['Term']
        
        if save_path is not None:
            GOBP_up.results.to_csv(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_UP.csv')
            
    GSEA_down_result = {}
    for i in range(0, len(top_ids)):  
        GOBP_down = gp.enrichr(gene_list=down_features_org2[i],
                               gene_sets=['GO_Biological_Process_2021'],
                               organism=species)
        GSEA_down_result[int(top_ids[i])] = GOBP_down.results
        
        GOBP_down.results['-log10_p_adj'] = -np.log10(GOBP_down.results['Adjusted P-value'])
        GOBP_down.results.index = GOBP_down.results['Term']
                
        if save_path is not None:
            GOBP_down.results.to_csv(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_DOWN.csv')
    
    return GSEA_up_result, GSEA_down_result

def highly_variable_features(df, n_top_features):
    """
    Identifies the most variable features in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing feature expression data. Columns should represent
        feature names, and rows should correspond to individual samples.
        
    n_top_features : int
        The number of top variable features to return.
    
    Returns
    -------
    list
        A list containing the names of the most variable features.
    """


    mean = df.mean(axis=0)
    variance = df.var(axis=0)
    df_fluctuation = pd.DataFrame({'mean': mean, 'variance': variance})
    df_fluctuation["residuals"] = np.log10(df_fluctuation["variance"]) - np.log10(df_fluctuation["mean"])
    
    # Fitting a linear regression model
    slope, intercept = np.polyfit(np.log10(df_fluctuation["mean"]), np.log10(df_fluctuation["variance"]), 1)
    df_fluctuation["fitted"] = intercept + slope * np.log10(df_fluctuation["mean"])
    
    # Subtract fitted values from the observed values
    df_fluctuation["dispersion"] = df_fluctuation["residuals"] - df_fluctuation["fitted"]
    
    # Selecting the most variable features
    df_fluctuation = df_fluctuation.sort_values(by='dispersion', ascending=False)
    hvf = df_fluctuation.head(n_top_features).index.tolist()
    
    return hvf