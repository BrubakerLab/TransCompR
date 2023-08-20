# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:49:37 2023

@author: RR
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.svm import LinearSVC, SVC
import warnings
from sklearn.linear_model import  LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from scipy.stats import ttest_ind
from .tools import  highly_variable_features
from .utils import  indices, validate_dataframes, validate_hvf_and_pcs, validate_classes

def TransCompR(organism1_data, organism2_data, organism2_classes, organism2_covar = None,non_cat_covar = [],
               n_pcs = 50, hvf = 0, use_interaction=False, penalty = 'l1', 
               l1_ratio = None, use_BO = False, BO_iter = 20, cv = 5,
               random_state = None, **kwargs):
    """
    Perform analysis and regression on input data from two organisms.
    
    Parameters
    ----------
    organism1_data : DataFrame
        A pandas DataFrame of shape (n_organism1, n_features) with matched feature columns 
        based on homology with organism2_data. Pre-processed data is expected.
    
    organism2_data : DataFrame
        A pandas DataFrame of shape (n_organism2, n_features) with matched feature columns 
        based on homology with organism1_data. Pre-processed data is expected.
    
    organism2_classes : Series or array-like
        Phenotypes for regression for organism2.
    
    organism2_covar : DataFrame, optional
        A DataFrame of shape (organism2, categorical_info). Categorical information for organism2.
        Dummy variables will be automatically generated, so they are not accepted. Default is None.
    
    n_pcs : int or str, optional
        Number of principal components to use. If 'max', maximum number of PCs will be used.
        Default is 50.
    
    hvf : int or list, optional
        Defines highly variable features. If 0, all features are used. If it's an integer, top 'hvf' features with 
        the largest residual in a mean-variance relationship regression will be used. If it's a list of feature symbols 
        for organism1, then that list will be used. Default is 0.
    
    use_interaction : bool, optional
        Whether to include interactions between TranscComps and covariates. Default is False.
    
    penalty : {'l1', 'l2', 'elasticnet', None}, optional
        The penalty type used for logistic regression. Default is 'l1'.
    
    l1_ratio : float, optional
        Parameter used only when penalty is 'elasticnet'. It should be between 0 and 1. Default is None.
    
    use_BO : bool, optional
        If True, use Bayesian Optimizer for selecting C in logistic regression. Default is False.
    
    BO_iter : int, optional
        Number of iterations for the Bayesian optimizer. Used only if use_BO is True. Default is 20.
    
    cv : int, optional
        Number of folds for cross-validation. Used only if use_BO is True. Default is 5.
    
    random_state : int or RandomState instance, optional
        Random state seed used for shuffling data when solver is one of {'sag', 'saga', 'liblinear'}. Default is None.
    
    Returns
    -------
    results : dict
        A dictionary with results and metrics from the regression analysis. Contains keys:
        - 'organism1_hvf': List of calculated highly variable features in organism1 data if 'hvf' is not None.
        - 'model': Logistic regression model for organism2 classes.
        - 'organism1_loadings': DataFrame (n_pcs, n_features) with loadings of organism1 features.
        ... [Include other keys similarly formatted]
    
    Notes
    -----
    Ensure that both organism data inputs are pre-processed and have matching features based on homology.
    """
    warnings.filterwarnings("ignore")
    
    n_features = organism1_data.shape[1]
    
    # Z-score normalize feature-matched organism1 and organism2 data matrices
    scaler = StandardScaler()
    organism2_Zdata = scaler.fit_transform(organism2_data.values)
    organism1_Zdata = scaler.fit_transform(organism1_data.values)
    
    # Train organism1 Data PCA Model
    if hvf == 0:
        hvf = organism1_data.columns.tolist()
    elif isinstance(hvf, int) and hvf > 0 and hvf <= organism1_data.shape[1]:
        hvf = highly_variable_features(organism1_data,hvf)   
    elif isinstance(hvf, list):
        hvf = hvf
    else:
        raise ValueError("If you want to use hvf, please pass an positive integer that is less than the total number of features in the dataset. Otherwise set False.")
    keep_id = np.where([i in hvf for i in organism1_data.columns.tolist()])
    organism1_data = organism1_data.iloc[:,keep_id[0]]
    organism1_Zdata = organism1_Zdata[:,keep_id[0]]
    organism2_data = organism2_data.iloc[:,keep_id[0]]
    organism2_Zdata = organism2_Zdata[:,keep_id[0]]

    
    if isinstance(n_pcs,str) and n_pcs == 'max':
        n_pcs = np.min(organism1_data.shape)
        print('Max n_pcs used. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) and n_pcs > np.min(organism1_data.shape): 
        n_pcs = np.min(organism1_data.shape)
        print('n_pcs exceeded the maximum. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) and n_pcs <= 0: 
        n_pcs = np.min(organism1_data.shape)
        print('n_pcs cannot be non-negative. Set n_pcs to '+str(n_pcs))
    elif isinstance(n_pcs,int) == False:
        raise ValueError("Invalid n_pcs argument.")

   
    pca = PCA(n_components = n_pcs, random_state= random_state)
    organism1_scores = pca.fit_transform(organism1_Zdata)
    organism1_loadings = pd.DataFrame(pca.components_)
    organism1_loadings.columns = organism1_data.columns
    #organism1_explained = pca.explained_variance_ratio_
    
    
    # Project organism2 data Into organism1 PCA
    organism2_transComps = scaler.fit_transform(np.dot(organism2_Zdata, organism1_loadings.T))
    organism2_transComps = pd.DataFrame(organism2_transComps, index = organism2_data.index)

    organism2_transComps_noZ = np.dot(organism2_Zdata, organism1_loadings.T)
    organism2_transComps_noZ = pd.DataFrame(organism2_transComps_noZ, index = organism2_data.index)     

    transCompR_table = pd.concat([organism2_transComps, pd.DataFrame(organism2_classes.T)], axis=1)
    transCompR_table_noZ = pd.concat([organism2_transComps_noZ, pd.DataFrame(organism2_classes.T)], axis=1)
    
    # Identify organism1 PCs predictive of organism2 classes
    # Assuming last column in table is the target variable for regression
    X = transCompR_table.iloc[:,:-1]
    X_noZ = transCompR_table_noZ.iloc[:,:-1]
    y = transCompR_table.iloc[:,-1]
    
    if organism2_covar is not None and isinstance(organism2_covar, pd.DataFrame):
       #Now we deal with covariates
        prefix = dict()
        noncat_covar = organism2_covar[non_cat_covar]
        noncat_covar_noZ = noncat_covar.copy()
        for i in noncat_covar.columns:
            continuous_covar = []
            for j in noncat_covar[i]:
                try:
                    continuous_covar.append(float(j))
                except:
                    continuous_covar.append(np.nan)
            noncat_covar[i] = scaler.fit_transform(np.array(continuous_covar).reshape(-1,1))
            noncat_covar_noZ[i] = np.array(continuous_covar)
            
        cat_covar = organism2_covar.iloc[:,~organism2_covar.columns.isin(non_cat_covar)]
        for i in cat_covar.columns:
            cat_covar[i] = cat_covar[i].astype('category')
            prefix[i] = 'dummy_'+i
        add_dummies = pd.get_dummies(cat_covar, prefix = prefix)
        add_dummies_noZ = pd.concat([add_dummies, noncat_covar_noZ],axis = 1)
        add_dummies = pd.concat([add_dummies, noncat_covar],axis = 1)

        # Adding interaction terms
        if use_interaction:
            for trans_col in organism2_transComps.columns:
                for dummy_col in add_dummies.columns:
                    interaction_col_name = f"interaction_{trans_col}_{dummy_col}"
                    X[interaction_col_name] = organism2_transComps[trans_col] * add_dummies[dummy_col]
                    X_noZ[interaction_col_name] = organism2_transComps_noZ[trans_col] * add_dummies_noZ[dummy_col]
                    
        X = pd.concat([X, add_dummies],axis = 1)
        X_noZ = pd.concat([X_noZ, add_dummies_noZ],axis = 1)
        
    X.columns = X.columns.astype(str)
    new_tcr_table = pd.concat([X,y],axis = 1)
    new_tcr_table = new_tcr_table.dropna()
    
    X = new_tcr_table.iloc[:,:-1]
    y = new_tcr_table.iloc[:,-1]
    
    X_noZ.columns = X_noZ.columns.astype(str)
    new_tcr_table = pd.concat([X_noZ,y],axis = 1)
    new_tcr_table = new_tcr_table.dropna()
    X_noZ = new_tcr_table.iloc[:,:-1]
    
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/cv, random_state=random_state)

    if penalty == 'l1':
        solver_name = 'liblinear'
    elif penalty == 'l2':
        solver_name = 'lbfgs'
    elif penalty == 'elasticnet':
        solver_name = 'saga'        
    elif penalty == 'None':
        solver_name = 'lbfgs'
    else:
        print("Invalid penalty argument. Use no penalty.")

    # Function to optimize
    C = 1.0
    
    if use_BO:
        def evaluate(C):
            model = LogisticRegression(**kwargs, C=C, penalty = penalty, solver=solver_name,random_state=random_state,l1_ratio=l1_ratio)
            return cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    
        # Bounded region of parameter space
        pbounds = {'C': (1e-6, 2)}
    
        optimizer = BayesianOptimization(
            f=evaluate,
            pbounds=pbounds,
            random_state=random_state,
        )
    
        optimizer.maximize(n_iter=BO_iter)
        
        best_params = optimizer.max['params']
        C = best_params['C']
    
    # Refit the best model to the entire dataset
    transCompR_mdl = LogisticRegression(**kwargs, penalty = penalty, solver=solver_name, C=C, random_state=random_state, l1_ratio = l1_ratio).fit(X_train, y_train)
    y_predict = transCompR_mdl.predict(X_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    
    scores = pd.DataFrame([accuracy, precision, recall]).T
    scores.columns = ['test accuracy', 'test precision', 'test recall']
    
    top_organism1 = [organism1_data.columns[np.argsort(organism1_loadings.iloc[i,:])].tolist() for i in range(0,n_pcs)] #ascending
    top_organism1_loadings = [organism1_loadings.values[i,np.argsort(organism1_loadings.iloc[i,:])] for i in range(0,n_pcs)]
    top_coef_top_features = pd.DataFrame({'PCs': [str(i) for i in range(1,n_pcs+1)],
                                   'coefs':transCompR_mdl.coef_[0][0:n_pcs],
                                   'sorted_organism1_features': top_organism1,
                                   'sorted_organism1_features_loadings': top_organism1_loadings,
                                   'organism1_explained': np.var(organism1_scores, axis = 0)/n_features,
                                   'organism2_explained': np.var(organism2_transComps_noZ, axis = 0)/n_features})
    
    if organism2_covar is not None and isinstance(organism2_covar, pd.DataFrame):
        indiv_accuracy = np.zeros(n_pcs)
        for i in range(0,n_pcs):
            indiv_X = X.iloc[:,i]
            if use_interaction:
                for dummy_col in add_dummies.columns:
                    interaction_col_name = f"interaction_{trans_col}_{dummy_col}"
                    indiv_X[interaction_col_name] = X.iloc[:,i] * add_dummies[dummy_col]
                indiv_X = pd.concat([indiv_X, add_dummies],axis = 1)
            else:
                indiv_X = pd.concat([indiv_X, add_dummies],axis = 1)
            
            indiv_model = LogisticRegression(**kwargs,penalty = penalty, solver=solver_name,random_state=random_state,l1_ratio = l1_ratio).fit(indiv_X.loc[X_train.index,:], y_train)
            indiv_accuracy[i] = accuracy_score(y_test, indiv_model.predict(indiv_X.loc[X_test.index,:]))
        top_coef_top_features['individual_predictivity'] = indiv_accuracy
    else:
        
        indiv_accuracy = np.zeros(n_pcs)
        for i in range(0,n_pcs):
            indiv_model = LogisticRegression(**kwargs,penalty = penalty, solver=solver_name,random_state=random_state,l1_ratio=l1_ratio).fit(X_train.iloc[:,i].values.reshape(-1,1), y_train)
            indiv_accuracy[i] = accuracy_score(y_test, indiv_model.predict(X_test.iloc[:,i].values.reshape(-1,1)))
          
    db_score = [davies_bouldin_score(X_noZ.iloc[:,i].values.reshape(-1,1),y) for i in range(0,n_pcs)]
    top_coef_top_features['davies_bouldin_score'] = db_score
    
    unique_categories = list(set(y))
    if len(unique_categories) != 2:
        raise ValueError("y should have exactly 2 unique categories for this calculation")
    
    category_1 = unique_categories[0]
    category_2 = unique_categories[1]
    
    # Calculate Calinski and Harabasz Score for each PC
    ch_scores = [calinski_harabasz_score(X_noZ.iloc[:,i].values.reshape(-1,1), y) for i in range(0, n_pcs)]
    top_coef_top_features['calinski_harabasz_score'] = ch_scores
    
    # Calculate Difference of Mean between two categories for each PC
    differences = [X_noZ[y == category_1].iloc[:,i].mean() - X_noZ[y == category_2].iloc[:,i].mean() for i in range(0, n_pcs)]
    top_coef_top_features['difference_of_means'] = differences
    
    # Calculate t-test p-value for difference of means between two categories for each PC
    p_values = []
    for i in range(0, n_pcs):
        category_1_data = X_noZ[y == category_1].iloc[:,i]
        category_2_data = X_noZ[y == category_2].iloc[:,i]
        t_stat, p_value = ttest_ind(category_1_data, category_2_data)
        p_values.append(p_value)
    top_coef_top_features['t_test_p_value'] = p_values
        
    all_regression_coeffs = pd.DataFrame()
    all_regression_coeffs['coefs'] = transCompR_mdl.coef_[0]
    all_regression_coeffs.index = X.columns.tolist()
    results = {
        'organism1_features': organism1_data.columns,
        'organism2_features': organism2_data.columns,
        'organism1_hvf' : hvf,
        'model': transCompR_mdl,
        'organism1_loadings': organism1_loadings,
        'organism1_scores': organism1_scores,
        'organism2_transComps': organism2_transComps_noZ,
        'organism2_classes': organism2_classes,
        'X': X,
        'X_noZ': X_noZ,
        'y': y,
        'training':[X_train,y_train],
        'testing': [X_test, y_test],
        'regression_terms': X.columns.tolist(), # Add the regression terms
        'all_regression_coeffs': all_regression_coeffs, # Store all regression coefficients
        'cross_validation_metrics': scores,
        'predictivity_summary': top_coef_top_features
    }
    return results