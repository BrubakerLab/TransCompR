# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:49:37 2023

@author: RR
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def TransCompR(human_data, human_classes, mouse_data):
    """
    INPUTS
    human_data:       pandas DataFrame of genes x patients
    mouse_data:       pandas DataFrame of genes x mice
    human_classes:    human phenotypes for regression
    
    OUTPUTS
    transCompR_mdl:   regression model
    human_transComps: projection of human data in mouse PC space
    mouse_loadings:   Loadings of proteins in mouse PCA model
    mouse_explained:  Percent variance explained by mouse PCs
    mouse_scores:     Mouse scores on mouse PCs. 
    """
      
    # Z-score normalize feature-matched mouse and human data matrices
    scaler = StandardScaler()
    human_Zdata = scaler.fit_transform(human_data.values.T)
    mouse_Zdata = scaler.fit_transform(mouse_data.values.T)

    # Train Mouse Data PCA Model
    pca = PCA()
    mouse_scores = pca.fit_transform(mouse_Zdata)
    mouse_loadings = pca.components_
    mouse_explained = pca.explained_variance_ratio_

    # Project Human data Into Mouse PCA
    human_transComps = np.dot(human_Zdata, mouse_loadings.T)

    # Create DataFrame for GLM Model
    transCompR_table = pd.concat([pd.DataFrame(human_transComps), pd.DataFrame(human_classes.T)], axis=1)

    # Identify mouse PCs predictive of human classes
    # Assuming last column in table is the target variable for regression
    X = transCompR_table.iloc[:,:-1]
    y = transCompR_table.iloc[:,-1]
    transCompR_mdl = LinearRegression().fit(X, y)

    return transCompR_mdl, human_transComps, mouse_loadings, mouse_explained, mouse_scores
