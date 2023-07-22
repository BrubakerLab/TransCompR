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
import TransCompR as tr
```

### 2. Applying the TransCompR Method

The main function in this package is `TransCompR`, which performs the cross-species transcriptomic comparison. You can apply this function on your data like this:

```
results = tr.TransCompR(human_df, mouse_df, human_classes)
```

In this function:

- `human_df` and `mouse_df` are pandas DataFrame where each row represents a sample and each column represents a gene. The entries are gene expression values. The value is expected to be properly preprocessed.

- `human_classes` is the binary class or condition of each sample in `human_df`. They should be the same length as the number of human subjects.

- `results` is a dictionary that contains the result of the analysis, including the calculated loadings, explained variance, and more.

You can extract these results for further analysis and visualization. For example, to extract the loadings, you could do:

```
loadings = results['loadings']
```
To explore the `TransCompR` framework thoroughly, please see the details below.

>#### Detailed Function Explanation: TransCompR
>**Inputs**
>
>- `mouse_data`: pandas DataFrame of n_mice x n_genes. Expected to be pre-processed and have matched gene columns with human_data based on homology.
>- `human_data`: pandas DataFrame of n_patients x n_genes. Expected to be pre-processed and have matched gene columns with mouse_data based on homology.
>- `human_classes`: human phenotypes for regression.
>- `human_covar`: pandas DataFrame of patients x categorical info (default: None). This function will automatically generate dummies, so dummy variables are not accepted.
>- `n_pcs`: positive int, 'max' (default: 50). When 'max' is input, maximum number of PCs will be used.
>- `hvg`: non-negative int (default: 0). When hvg = 0, all genes are used. Otherwise, top x genes with the largest residual in a mean-variance relationship regression will be used.
>- `**kwargs`: Additional keywords pass to the sklearn.linear_model.LogisticRegression. See [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for proper penalty and model selection.
>- 
>**Outputs**
>
>The function returns a dictionary `results` that contains the major outputs and information used in downstream analysis.
>
>- `results['mouse_hvg']`: list of calculated highly variable genes (hvg) in mouse data if argument 'hvg' is not None
>- `results['model']`: sklearn logistic regression model for human classes regressed over human features. Also include covariates if covariates are used.
>- `results['mouse_loadings']`: pandas DataFrame of n_pcs x n_genes. Loadings of mouse genes on mouse PCs.
>- `results['mouse_scores']`: Array of float64 of n_mice x n_pcs. score of mouses on mouse PCs.
>- `results['human_transComps']`: pandas DataFrame of n_patients x n_pcs. Porjected scores of human on mouse PCs.
>- `results['human_classes']`: Copy of human phenotypes for regression. Stored here for downstream visualizations.
>- `results['training']`: list of pandas DataFrame. The first entry is X_train, a dataframe of n_patients x n_features used in model training. Dummies of covariates >are appended after the PC columns if covariates are used. The second entry is y_train.
>- `results['testing']`: list of pandas DataFrame. The first entry is X_test, a dataframe of n_patients x n_features used in model testing. Dummies of covariates >are appended after the PC columns if covariates are used. The second entry is y_test.
>- `results['cross_validation_metrics']`: pandas DataFrame contains the accuracy, precision, and recall of the logistic regression model on the testing data.
>- `results['predictivity_summary']`: pandas DataFrame contains information about the predictivity of each mouse PC on human_classes.
>   - rows: PCs index in an ascending order.
>    - columns:
>      - `'PCs'`: str. Name of each PC. Essentially it is the index + 1.
>      - `'coefs'`: float64. Coefficients of each PC in the logistic regression. If human_covar is not None, this will be the coefficients of PCs while taking the covariate effect into consideration.
>      - `'sorted_mouse_genes'`: list. Names of mouse genes sorted by their loadings on each PC in an ascending order.
>      - `'sorted_mouse_genes_loadings'`: list. the corresponding loadings of the `'sorted_mouse_genes'`.
>      - `'mouse_explained'`: float 64. Percentage of the total variance in mouse data explained by mouse PCs.
>      - `'human_explained'`: float 64. Percentage of the total variance in human data explained by mouse PCs
>      - `'individual_predictivity'`: float 64. The accuracy on the testing data when only the current PC (and input covariates) are in the regression.
>      - `'davies_bouldin_score'`: float 64. The Davies-Bouldin score of human classes on the current PC.

### 3. Visualize the result using `plot.var_explained`, `plot.partition`, `plot.gsea`, `plot.loadings`

`TransCompR` provided 4 functions to visualize the results.

#### **1. `plot.var_explained`: Visualize the total variance explained in mouse and human data by mouse principal components (PCs).**

  For example:
  ```
  tr.plot.var_explained(results, n_pcs = 20)
  ```
  ...will plot two bar plots showing the total variance explained in mouse and human data by the first 20 mouse PCs.

>> **Args:**
>> 
>> - **results (dict):** Results dictionary with key `'predictivity_summary'` containing keys `'human_explained'` and `'mouse_explained'` referring to the percentage variance explained by mouse PCs in human and mouse data, respectively.
>> 
>> - **n_pcs (int, optional):** Number of PCs to be included in the visualization. Default to 20.
>> 
>> - **save (str, optional):** Path to save the figure. If None, the figure will not be saved. Default to None.
>> 
>> - **return_figures (bool, optional):** Whether to return the `matplotlib.figure.Figure instance`. If False, nothing will be returned. Default to False.
>> 
>> - **plot_kwargs1 (dict, optional):** Additional keyword arguments to pass to `matplotlib.pyplot.bar` for the mouse variance plot. Default to None.
>> 
>> - **plot_kwargs2 (dict, optional):** Additional keyword arguments to pass to `matplotlib.pyplot.bar` for the human variance plot. Default to None.
>> 
>> **Returns:**
>> 
>> - **fig (matplotlib.figure.Figure, optional):** A `matplotlib.figure.Figure` instance if return_figures is True, else None.


#### **2.`plot.partition`: visualizes the Kernel Density Estimation (KDE) of human samples based on top Principal Components (PCs) that can best separate two human classes**
PC's performance in separating 2 classes is evaluated by one of the following 3 metrics: metrics: `'db'` (Davies Bouldin Score), `'coefs'` (Absolute coefficients in the overall regression model), or `'indiv acc'` (Individual Accuracy).

  For example:
  ```
  tr.plot.partition(results,  metric = 'db', top_n_pcs = 5, save_path = 'D://my_folder//figure')
  ```
  ...will generate the Gaussian KDE filled plots of the top 5 PCs that can best separate the human classes based on Davies Bouldin Score and save the file to `'D://my_folder//figure'`

  Or, when `pcs` are used:

  ```
  tr.plot.partition(results,  metric = 'db', pcs = [0,1,10], save_path = 'D://my_folder//figure')
  ```
  ...will generate the Gaussian KDE filled plots of PC1, PC2, and PC11, report Davies Bouldin Score, and save the figures to `'D://my_folder//figure'`
  
>> **Args:**
>> 
>> - **results(dict):** A dictionary containing the results of previous computations. Expected to contain the key `'predictivity_summary'` and `'human_transComps'`.
>>   
>> - **metric (str):** The metric to be used for selecting top principal components. Accepts `'db'`, `'coefs'`, `'indiv acc'`. Default is `'db'`.
>>   
>> - **top_n_pcs (int):** The top N principal components to be considered. Default is 5.
>>
>> - **pcs (list of int)**: When specified, the specific pc is plotted instead of top pcs. Please be awared that python indexing starts from 0, so [0] would plot the PC1 and [1] would plot the PC2.
>> 
>> - **save_path (str):** The directory path where generated figures should be saved. If None, figures will not be saved.
>>   
>> - **return_figures (bool):** If True, the figures generated by the function will be returned. Default is False.
>>   
>> - **fontsize (int):** Font size for the plot title and labels. Default is 12.
>>   
>> - **fontweight (str):** Font weight for the plot title and labels. Default is `'bold'`.
>>   
>> - **facecolor (list):** List of colors to be used for the two classes in the plot. Default is `["blue", "orange"]`.
>>   
>> - **kwargs (dict):** Additional keyword arguments to pass to the bar plot function.
>>
>> **Returns:**
>> 
>> - **figures (list of matplotlib.figure.Figure, optional):** A list of figures if return_figures is set to True. Otherwise, it doesn't return anything.

#### **3. `plot.gsea`:performs Gene Set Enrichment Analysis (GSEA) on genes that have the largest loadings on the principal components (PCs) that best separate two human classes.**

PC's performance in separating 2 classes is evaluated by one of the following 3 metrics: metrics: `'db'` (Davies Bouldin Score), `'coefs'` (Coefficients in the overall regression model), or `'indiv acc'` (Individual Accuracy).

  For example:
  ```
  GESA_up_result, GESA_down_result = tr.plot.gsea(results, metric = 'db',  top_n_pcs = 5, top_n_genes = 100, save_path = 'D://my_folder//figure')
  ```
  ...will inquire the top 100 genes with largest positive loadings and top 100 genes with largest negative loadings in the 5 PCs that can best separate the human classes based on Davies Bouldin Score and save the file to `'D://my_folder//figure'`. The direction (which human class is more positive on the current PC) is determined from the mean PC score in each class and is annotated in each plots' title. Let's say on a PC on which healthy controls have more positive score than diseased patients, genes with the most positive loadings on this PC may has inverse correlation with disease activities. For each top PC, you will get two GSEA visualization, 1 for positive loadings genes' pathways, and 1 for negative loadings genes' pathways.

 Or, when `pcs` are used:

  ```
 GESA_up_result, GESA_down_result = tr.plot.gsea(results, pcs = [0,1,10], save_path = 'D://my_folder//figure')
  ```
  ...the function will perform GSEA on the top 100 genes with largest positive loadings and top 100 genes with largest negative loadings in PC1, PC2, and PC11, and save the figures to `'D://my_folder//figure'`

>> **Args:**
>> 
>> - **results(dict):** A dictionary containing the results of previous computations. Expected to contain the key `'predictivity_summary'` and `'human_transComps'`.
>>   
>> - **metric (str):** The metric to be used for selecting top principal components. Accepts `'db'`, `'coefs'`, `'indiv acc'`. Default is `'db'`.
>>   
>> - **top_n_pcs (int):** The top N principal components to be considered. Default is 5.
>>
>> - **pcs (list of int)**: When specified, the specific pc is plotted instead of top pcs. Please be awared that python indexing starts from 0, so [0] would plot the PC1 and [1] would plot the PC2.
>> 
>> - **top_n_genes (int):** The top N genes on both directions to be inquired in database for GSEA. Default is 100.
>>
>> - **n_go (int):** The number of gene ontology terms to consider. Default is 10.
>>   
>> - **save_path (str):** The directory path where generated figures should be saved. If None, figures will not be saved.
>>   
>> - **return_figures (bool):** If True, the figures generated by the function will be returned. Default is False.
>>   
>> - **kwargs (dict):** Additional keyword arguments to pass to the bar plot function.
>>
>> **Returns:**
>> - **GESA_up_result (dict)**: A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for upregulated genes.
>>   
>> - **GESA_down_result (dict)**: A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for downregulated genes.
>>   
>> - **figures_up (list of matplotlib.figure.Figure, optional):** A list of figures shows the GSEA of positive loadings genes on PCs if return_figures is set to True. Otherwise, it doesn't return anything.
>>   
>> - **figures_down (list of matplotlib.figure.Figure, optional):** A list of figures shows the GSEA of negative loadings genes on PCs if return_figures is set to True. Otherwise, it doesn't return anything.
  
#### **4. `plot.loadings`: visualize loadings of mouse genes on 1 (strip plot) or 2 (scatter plot) mouse principal components.**

The numberic values of each genes' loadings can be found at `results['mouse_loadings']`(unsorted) and `results['predictivity_summary']['sorted_mouse_genes']`(sorted), this function provides a quick way to visualize gene loadings. For single PC, top N genes with the largest loadings on both direction are highlighted. For 2 PCs, top N genes with the the largest summation of the absolute values of 2 PCs are highlighted. 

Example:
```
tr.plot.loadings(results, [1,[1,2]], top_n_genes = 10)
```
...will 1) generate a boxplot-stacked strip plot of gene loadings on PC2, with top 10 genes with the largest loadings on both direction highlighted and labeled; 2) generate a scatter plot on which PC2 is the x axis and PC 3 is the y axis, with top 10 genes with the the largest summation of the absolute values of 2 PCs highlighted.

>> **Args:**
>> 
>> - **results(dict):** A dictionary containing the results of previous computations. Expected to contain the key `'predictivity_summary'` and `'human_transComps'`.
>>   
>> - **pcs (list of int)**: When specified, the specific pc is plotted instead of top pcs. Please be awared that python indexing starts from 0, so [0] would plot the PC1 and [1] would plot the PC2.
>> 
>> - **top_n_genes (int):** The top N genes on both directions to be inquired in database for GSEA. Default is 20.
>> 
>> - **save_path (str):** The directory path where generated figures should be saved. If None, figures will not be saved.
>>   
>> - **return_figures (bool):** If True, the figures generated by the function will be returned. Default is False.
>>   
>> - **kwargs (dict):** Additional keyword arguments to pass to the bar plot function.
>>
>> **Returns:**
>>
>> - **figures (list of matplotlib.figure.Figure, optional):** A list of figures if return_figures is set to True. Otherwise, it doesn't return anything.
