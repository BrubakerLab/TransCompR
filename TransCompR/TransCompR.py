# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:49:37 2023

@author: RR
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, davies_bouldin_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import  LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization

from .tools import  highly_variable_genes
from .utils import  indices, validate_dataframes, validate_hvg_and_pcs, validate_classes


def TransCompR(mouse_data,human_data, human_classes, human_covar = None, n_pcs = 50, hvg = 0, **kwargs):
    """
    INPUTS
    mouse_data:     pandas DataFrame of n_mice x n_genes. Expected to be pre-processed and have matched gene columns
                    with human_data based on homology.
    human_data:     pandas DataFrame of n_patients x n_genes. Expected to be pre-processed and have matched gene columns
                    with mouse_data based on homology.
    human_classes:  human phenotypes for regression.
    human_covar:    pandas DataFrame of patients x categorical info (default: None). This function will automatically generate dummies
                    so dummy variables are not accepted.
    n_pcs:          positive int, 'max' (default: 50). When 'max' is input, maximum number of PCs will be used.
    hvg:            non-negative int (default: 0). When hvg = 0, all genes are used. Otherwise, top x genes with the
                    largest residual in a mean-variance relationship regression will be used.
    
    OUTPUTS
    results:         dictionary that contains the major outputs and information used in downstream analysis
        -keys:
            'mouse_hvg' :               list of calculated highly variable genes (hvg) in mouse data if argument 'hvg' is not None
            'model':                    sklearn logistic regression model for human classes regressed over human features. 
                                        Also include covariates if covariates are used. 
            'mouse_loadings':           pandas DataFrame of n_pcs x n_genes. Loadings of mouse genes on mouse PCs.
            'mouse_scores':             Array of float64 of n_mice x n_pcs. score of mouses on mouse PCs.
            'human_transComps':         pandas DataFrame of n_patients x n_pcs. Porjected scores of human on mouse PCs.
            'human_classes':            Copy of human phenotypes for regression. Stored here for downstream visualizations.
            'training':                 list of pandas DataFrame. The first entry is X_train, a dataframe of 
                                        n_patients x n_features used in model training. Dummies of covariates are
                                        appended after the PC columns if covariates are used. The second entry is y_train.                                        
            'testing':                  list of pandas DataFrame. The first entry is X_test, a dataframe of 
                                        n_patients x n_features used in model testing. Dummies of covariates are
                                        appended after the PC columns if covariates are used. The second entry is y_test. 
            'cross_validation_metrics': pandas DataFrame contains the accuracy, precision, and recall of the logistic 
                                        regression model on the testing data.
            'predictivity_summary':     pandas DataFrame contains information about the predictivity of each mouse PC
                                        on human_classes.
                                        -rows:                              PCs index in an ascending order.
                                        -columns:
                                            'PCs':                          str. Name of each PC. Essentially it is the index + 1 
                                            'coefs':                        float64. Coefficients of each PC in the logistic regression.
                                                                            If human_covar is not None, this will be
                                                                            coefficients of PCs while taking the covariate effect into consideration.                                                     
                                            'sorted_mouse_genes':           list. Names of mouse genes sorted by their loadings on each PC in an ascending order
                                            'sorted_mouse_genes_loadings':  list. the corresponding loadings of the 'sorted_mouse_genes' in the same order.
                                            'mouse_explained':              float 64. Percentage of total variance in mouse data explained by mouse PCs.
                                            'human_explained':              float 64. Percentage of total variance in human data explained by mouse PCs
                                            'individual_predictivity':      float 64. The accuracy on the testing data when only the current PC (and input covariates) are in the regression.
                                            'davies_bouldin_score':         float 64. The Davies-Bouldin score of human classes on current PC.
    """
    
    validate_dataframes(human_data, mouse_data, human_covar)
    validate_hvg_and_pcs(hvg, n_pcs, mouse_data)
    human_classes = validate_classes(human_classes)
    
    n_genes = mouse_data.shape[1]
    
    # Z-score normalize feature-matched mouse and human data matrices
    scaler = StandardScaler()
    human_Zdata = scaler.fit_transform(human_data.values)
    mouse_Zdata = scaler.fit_transform(mouse_data.values)
    
    # Train Mouse Data PCA Model
    if hvg == 0:
        HVG = mouse_data.columns.tolist()
    elif isinstance(hvg, int) and hvg > 0 and hvg <= mouse_data.shape[1]:
        HVG = highly_variable_genes(mouse_data,hvg)
        
        mouse_data = mouse_data.loc[:,HVG]
        mouse_Zdata = mouse_Zdata[:,indices(mouse_data.columns.isin(HVG),True)]
        human_data = human_data.loc[:,[i[0]+i[1:].upper() for i in HVG]]
        human_Zdata = human_Zdata[:,indices(human_data.columns.isin([i[0]+i[1:].upper() for i in HVG]),True)]
    else:
        raise ValueError("If you want to use HVG, please pass an positive integer that is less than the total number of genes in the dataset. Otherwise set False.")
    
    print(n_pcs)
    if isinstance(n_pcs,str) and n_pcs == 'max':
        n_pcs = np.min(mouse_data.shape)
        print('Max n_pcs used. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) and n_pcs > np.min(mouse_data.shape): 
        n_pcs = np.min(mouse_data.shape)
        print('n_pcs exceeded the maximum. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) and n_pcs <= 0: 
        n_pcs = np.min(mouse_data.shape)
        print('n_pcs cannot be non-negative. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) == False:
        raise ValueError("Invalid n_pcs argument.")

   
    pca = PCA(n_components = n_pcs)
    mouse_scores = pca.fit_transform(mouse_Zdata)
    mouse_loadings = pd.DataFrame(pca.components_)
    mouse_loadings.columns = mouse_data.columns
    mouse_explained = pca.explained_variance_ratio_
    
    
    # Project Human data Into Mouse PCA
    human_transComps = np.dot(human_Zdata, mouse_loadings.T)
    human_transComps = pd.DataFrame(human_transComps, index = human_data.index)

    transCompR_table = pd.concat([human_transComps, pd.DataFrame(human_classes.T)], axis=1)


    # Identify mouse PCs predictive of human classes
    # Assuming last column in table is the target variable for regression
    X = transCompR_table.iloc[:,:-1]
    y = transCompR_table.iloc[:,-1]
    
    if human_covar is not None and isinstance(human_covar, pd.DataFrame):
        #Now we deal with covariates
        prefix = dict()
        for i in human_covar.columns:
            human_covar[i] = human_covar[i].astype('category')
            prefix[i] = 'dummy_'+i+'_'
        add_dummies = pd.get_dummies(human_covar, prefix = prefix)
        X = pd.concat([X, add_dummies],axis = 1)
    
    X.columns = X.columns.astype(str)
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to optimize
    def evaluate(C):
        model = LogisticRegression(**kwargs, C=C)
        model.fit(X_train, y_train)
        return cross_val_score(model, X_train, y_train, cv=KFold(n_splits=5), scoring="accuracy").mean()

    # Bounded region of parameter space
    pbounds = {'C': (1e-6, 2)}

    optimizer = BayesianOptimization(
        f=evaluate,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(n_iter=20)
    
    best_params = optimizer.max['params']

    # Refit the best model to the entire dataset
    transCompR_mdl = LogisticRegression(**kwargs, C=best_params['C']).fit(X, y)

    # Cross validation on the whole data
    cv_scores_accuracy = cross_val_score(transCompR_mdl, X_test, y_test, cv=5, scoring=make_scorer(accuracy_score)).mean()
    cv_scores_precision = cross_val_score(transCompR_mdl, X_test, y_test, cv=5, scoring=make_scorer(precision_score, average='macro')).mean()
    cv_scores_recall = cross_val_score(transCompR_mdl, X_test, y_test, cv=5, scoring=make_scorer(recall_score, average='macro')).mean()

    scores = pd.DataFrame([cv_scores_accuracy, cv_scores_precision, cv_scores_recall]).T
    scores.columns = ['test accuracy', 'test precision', 'test recall']
    
    top_mouse = [mouse_data.columns[np.argsort(mouse_loadings.iloc[i,:])].tolist() for i in range(0,n_pcs)] #ascending
    top_mouse_loadings = [mouse_loadings.values[i,np.argsort(mouse_loadings.iloc[i,:])] for i in range(0,n_pcs)]
    top_coef_top_genes = pd.DataFrame({'PCs': [str(i) for i in range(1,n_pcs+1)],
                                   'coefs':transCompR_mdl.coef_[0][0:n_pcs],
                                   'sorted_mouse_genes': top_mouse,
                                   'sorted_mouse_genes_loadings': top_mouse_loadings,
                                   'mouse_explained': np.var(mouse_scores, axis = 0)/n_genes,
                                   'human_explained': np.var(human_transComps, axis = 0)/n_genes})
    
    if human_covar is not None and isinstance(human_covar, pd.DataFrame):
        indiv_accuracy = np.zeros(n_pcs)
        for i in range(0,n_pcs):
            indiv_model = LogisticRegression(**kwargs).fit(pd.concat([X_train.iloc[:,i], X_train.iloc[:,n_pcs:]],axis = 1), y_train)
            indiv_accuracy[i] = cross_val_score(indiv_model,pd.concat([X_test.iloc[:,i], X_test.iloc[:,n_pcs:]],axis = 1), y_test, cv=5, scoring=make_scorer(accuracy_score)).mean()
        top_coef_top_genes['individual_predictivity'] = indiv_accuracy
    else:
        
        indiv_accuracy = np.zeros(n_pcs)
        for i in range(0,n_pcs):
            indiv_model = LogisticRegression(**kwargs).fit(X_train.iloc[:,i].values.reshape(-1,1), y_train)
            indiv_accuracy[i] = cross_val_score(indiv_model,X_test.iloc[:,i].values.reshape(-1,1), y_test, cv=5, scoring=make_scorer(accuracy_score)).mean()
        top_coef_top_genes['individual_predictivity'] = indiv_accuracy
          
    db_score = [davies_bouldin_score(X.iloc[:,i].values.reshape(-1,1),y) for i in range(0,n_pcs)]
    top_coef_top_genes['davies_bouldin_score'] = db_score
    
    results = {
        'mouse_hvg' : HVG,
        'model': transCompR_mdl,
        'mouse_loadings': mouse_loadings,
        'mouse_scores': mouse_scores,
        'human_transComps': human_transComps,
        'human_classes': human_classes,
        'training':[X_train,y_train],
        'testing': [X_test, y_test],
        'cross_validation_metrics': scores,
        'predictivity_summary': top_coef_top_genes
    }

    return results
