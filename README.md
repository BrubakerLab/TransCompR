# TransCompR

TransCompR is a Python package that enables cross-species transcriptomic comparison and visualization. This package includes a main function `TransCompR` to conduct a comparative analysis between human and mouse datasets, as well as several auxiliary functions to visualize and interpret the results.
(https://www.science.org/doi/10.1126/scisignal.aay3258)

## Installation
```
pip install TransCompR
```

## Usage

### 1. Importing the Package

After installing the package, you can import it in your Python script as follows:

```
import TransCompR as tcr
```

### 2. Applying the TransCompR Method

Check out our [tutorial](https://github.com/BrubakerLab/TransCompR/tree/main/tutorial). I'm still working on the documentation webpage, but below is the API if it helps.

### TransCompR Function

`TransCompR` is a function to perform analysis and regression on input data from two organisms.

```python
def TransCompR(organism1_data, organism2_data, organism2_classes, organism2_covar = None,non_cat_covar = [],
               n_pcs = 50, hvf = 0, use_interaction=False, penalty = 'l1', 
               l1_ratio = None, use_BO = False, BO_iter = 20, cv = 5,
               random_state = None, **kwargs):
```

#### Parameters:

- **organism1_data (DataFrame):**  
  A pandas DataFrame of shape (n_organism1, n_features) with matched feature columns based on homology with organism2_data. Pre-processed data is expected.

- **organism2_data (DataFrame):**  
  A pandas DataFrame of shape (n_organism2, n_features) with matched feature columns based on homology with organism1_data. Pre-processed data is expected.

- **organism2_classes (Series or array-like):**  
  Phenotypes for regression for organism2.
  
- **hvf (int or list, optional):**  
  Defines highly variable features. If 0, all features are used. If it's an integer, top 'hvf' features with the largest residual in a mean-variance relationship regression will be used. If it's a list of feature symbols for organism1, then that list will be used. Default is 0.

- **use_interaction (bool, optional):**  
  Whether to include interactions between TranscComps and covariates. Default is False.

- **penalty ({'l1', 'l2', 'elasticnet', None}, optional):**  
  The penalty type used for logistic regression. Default is 'l1'.

- **l1_ratio (float, optional):**  
  Parameter used only when penalty is 'elasticnet'. It should be between 0 and 1. Default is None.

- **use_BO (bool, optional):**  
  If True, use Bayesian Optimizer for selecting C in logistic regression. Default is False.

- **BO_iter (int, optional):**  
  Number of iterations for the Bayesian optimizer. Used only if use_BO is True. Default is 20.

- **cv (int, optional):**  
  Number of folds for cross-validation. Used only if use_BO is True. Default is 5.

- **random_state (int or RandomState instance, optional):**  
  Random state seed used for shuffling data when solver is one of {'sag', 'saga', 'liblinear'}. Default is None.

#### Returns:

**results**: `dict`
  A dictionary containing major outputs and information used in downstream analysis. The keys are:
  
  - **organism1_hvf**: `list`
    Calculated highly variable features (hvf) in organism1 data if 'hvf' is not None.

  - **model**: `sklearn.model`
    Logistic regression model for organism2 classes regressed over organism2 features. 
      Includes covariates if provided.

  - **organism1_loadings**: `pd.DataFrame`
    Dimensions: n_pcs x n_features. Loadings of organism1 features on organism1 PCs.

  - **organism1_scores**: `ndarray`
    Dimensions: n_organism1 x n_pcs. Scores of organisms on organism1 PCs.

  - **organism2_transComps**: `pd.DataFrame`
    Dimensions: n_organism2 x n_pcs. Projected scores of organism2 on organism1 PCs.

  - **organism2_classes**: `array-like`
    Copy of organism2 phenotypes for regression.

  - **X**: `pd.DataFrame`
    The matrix used for regression with all PCs and dummies/interaction terms. It is standardized 
      so that the covariates and interaction terms can be penalized.

  - **X_noZ**: `pd.DataFrame`
    The non-standardized matrix used for regression with all PCs and dummies/interaction terms.

  - **y**: `ndarray`
    Actual array of organism2 classes used in regression.

  - **training**: `list of pd.DataFrame`
    The first entry is X_train, a dataframe of n_organism2 x n_features used in model training. 
      Dummies of covariates are appended after the PC columns if they are used. 
      The second entry is y_train.

  - **testing**: `list of pd.DataFrame`
    The first entry is X_test, a dataframe of n_organism2 x n_features used in model testing. 
      Dummies of covariates are appended after the PC columns if they are used. 
      The second entry is y_test.

  - **cross_validation_metrics**: `pd.DataFrame`
    Contains accuracy, precision, and recall of the logistic regression model on the testing data.

  - **regression_terms**: `list of str`
    Names of the variables that were used in the regression.

  - **all_regression_coeffs**: `ndarray`
    Coefficients of all variables that were in the regression.

  - **predictivity_summary**: `pd.DataFrame`
    - Contains information about the predictivity of each organism1 PC on organism2_classes. The columns are:
      - **PCs**: Name of each PC. Essentially the index + 1.
      - **coefs**: Coefficients of each PC in the logistic regression.
      - **sorted_organism1_features**: Names of organism1 features sorted by their loadings on each PC.
      - **sorted_organism1_features_loadings**: Corresponding loadings of the 'sorted_organism1_features'.
      - **organism1_explained**: Percentage of total variance in organism1 data explained by organism1 PCs.
      - **organism2_explained**: Percentage of total variance in organism2 data explained by organism1 PCs.
      - **individual_predictivity**: Accuracy on the testing data when only the current PC (and input covariates) are used.
      - **davies_bouldin_score**: The Davies-Bouldin score of organism2 classes on the current PC.
      - **calinski_harabasz_score**: The Calinski-Harabasz Score of organism2 classes on the current PC.
      - **difference_of_means**: Difference in the mean PC score of organism2 classes on the current PC.
      - **t_test_p_values**: The p-values of t-test on the PC scores of organism2 classes.

