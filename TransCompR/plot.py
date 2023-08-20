# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:44:13 2023

@author: Ran
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.neighbors import KernelDensity
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from adjustText import adjust_text
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import ttest_ind
from .tools import  highly_variable_features
from .utils import  indices, validate_dataframes, validate_hvf_and_pcs, validate_classes

def loadings(results, pcs, organism1_name='organism1', top_n_features=20, save_path=None, 
             return_figures=False, figsize=(8,8), fontsize=20, fontweight='bold',
             textsize=8, textcolor='red', textweight='bold', stripdotsize=3, 
             highlightcolor='orange', jitter=0.3, scattersize=5):
    """
    Visualize loadings of features on each principal component for a given organism.
    
    Parameters
    ----------
    results : dict
        Results dictionary with a key such as '<organism>_loadings' that contains a DataFrame of feature loadings.
        
    pcs : list or int
        A list or an integer indicating which principal components to plot.
    
    top_n_features : int
        Number of top features to be highlighted on each PC.
    
    organism1_name : str, optional
        Name of organism1. Defaults to 'organism1'. Note: PCA is done on organism 1, so technically the loadings
        are an attribute of the organism1's genes.
    
    save_path : str, optional
        Path to save the figures. If None, figures will not be saved. Default is None.
    
    return_figures : bool, optional
        If True, returns the matplotlib.figure.Figure instances. Default is False.
    
    figsize : tuple, optional
        Size of the figure. Default is (8,8).
    
    fontsize : int, optional
        Font size for plot labels and title. Default is 30.
    
    fontweight : str, optional
        Font weight for plot labels and title. Default is 'bold'.
    
    textsize : int, optional
        Text size for annotations. Default is 8.
    
    textcolor : str, optional
        Text color for annotations. Default is 'red'.
    
    textweight : str, optional
        Text weight for annotations. Default is 'bold'.
    
    stripdotsize : int, optional
        Size of the dots in the strip plot. Default is 3.
    
    highlightcolor : str, optional
        Color to highlight the outliers. Default is 'orange'.
    
    jitter : float, optional
        Amount of jitter (only along the categorical axis) to apply. This can be useful when many points overlap, making it easier to see the distribution. Default is 0.3.
    
    scattersize : int, optional
        Size of the dots in the scatter plot. Default is 5.
    
    Returns
    -------
    figures : list or None
        List of matplotlib.figure.Figure instances if return_figures is True, else None.
    
    Raises
    ------
    ValueError
        If pcs is not an integer or a list of 2 integers.
    """

    
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.linewidth': 2})
    organism_loadings = results['organism1_loadings']

    figures = []
    
    for i in pcs:
        if isinstance(i,int):
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data = organism_loadings.iloc[i,:].T.sort_values()

            outliers = pd.concat([data.head(int(np.ceil(top_n_features/2))),data.tail(int(np.floor(top_n_features/2)))])
            insiders = data[int(np.ceil(top_n_features/2)):-int(np.floor(top_n_features/2))]

            path_collection = sns.stripplot(data=outliers, ax=ax, color=highlightcolor, size=stripdotsize*2, jitter=jitter).collections[0]
            coords = path_collection.get_offsets()
            xcoord= [coords[i][0] for i in range(0,len(outliers))]
            ycoord= [coords[i][1] for i in range(0,len(outliers))]

            texts = []
            for j, txt in enumerate(outliers.index):
                texts.append(ax.text(xcoord[j], ycoord[j], txt, size=textsize, color=textcolor, weight=textweight))
        
            sns.boxplot(data=data, ax=ax, fliersize=0, width=jitter*2)
            sns.stripplot(data=insiders, ax=ax, color=(0.5, 0.5, 0.5, 0.6), size=stripdotsize, alpha=0.3, jitter=jitter, zorder=1)

            ax.set_xlabel(f'{organism1_name}-derived PC ' + str(i + 1), size=fontsize, weight=fontweight)
            ax.set_ylabel('Loadings', size=fontsize, weight=fontweight)
            ax.set_xticks([0])
            ax.set_xticklabels('')
            adjust_text(texts)
            figures.append(fig)
            
        elif isinstance(i, list) and all([isinstance(j,int) for j in i]):
            if len(i) == 2:
                data = organism_loadings.iloc[i,:].T
                data['abs_sum'] = abs(data[i[0]]) + abs(data[i[1]])
                plot_entries = data.sort_values(by='abs_sum', ascending=False).head(top_n_features)
                labels = plot_entries.index.tolist()
                plot_entries = plot_entries.reset_index()
                
                fig, ax = plt.subplots(figsize=figsize)
    
                # move the left and bottom spines to x=0 and y=0, respectively.
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
    
                # remove the right and top spines
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
    
                # remove the ticks on the top and right axes
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                
                ax.scatter(data.iloc[:,0], data.iloc[:,1], s=scattersize*0.5, alpha=0.2, color='grey')
                ax.scatter(plot_entries.iloc[:,1], plot_entries.iloc[:,2], s=scattersize*2, color=highlightcolor)

                texts = []
                for j, txt in enumerate(labels):
                    texts.append(ax.text(plot_entries.iloc[j,1], plot_entries.iloc[j,2], txt, size=textsize, color=textcolor, weight=textweight))
                adjust_text(texts)
                
                ax.text(-0.05, 0.5, organism1_name+'-derived PC'+str(i[0]+1), va='center', ha='right', size=fontsize, transform=ax.transAxes, weight=fontweight,rotation='vertical')
                ax.text(0.5, 0, organism1_name+'-derived PC'+str(i[1]+1), va='top', ha='center', size=fontsize, transform=ax.transAxes, weight=fontweight)
                figures.append(fig)
            else:
                raise ValueError("Only support 2D plot.")
        else:
            raise ValueError("Invalid pcs argument.")
    
        if save_path is not None:
            fig.savefig(save_path+'Loadings_PC'+str(np.array(i)+1)+'.png')

    if return_figures:
        return figures

def var_explained(results, organism1_name='organism1', organism2_name='organism2',
                  n_pcs=20,  facecolor=['#00AFB9','#F07167'], save_path=None, 
                  return_figures=False, **kwargs):
    """
    Visualize the total variance explained in data from two organisms by the principal components (PCs) of organism1.
    
    Parameters
    ----------
    results : dict
        A dictionary containing results with a key 'predictivity_summary'. This key should have a sub-dictionary with 
        'organism2_explained' and 'organism1_explained' keys. These represent the percentage variance explained by 
        organism1's PCs in organism2 and organism1 data, respectively.
    
    organism1_name : str, optional
        Name of the first organism for visualization purposes. Defaults to 'organism1'.
    
    organism2_name : str, optional
        Name of the second organism for visualization purposes. Defaults to 'organism2'.
    
    n_pcs : int, optional
        Number of PCs to include in the visualization. Defaults to 20.
    
    facecolor : list, optional
        Colors for the two classes in the plot. Defaults to ['#D2493A','#28548f'].
    
    save_path : str, optional
        Path where the figure should be saved. If None, the figure won't be saved. Defaults to None.
    
    return_figures : bool, optional
        If True, returns the matplotlib.figure.Figure instance. Defaults to False.
    
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the matplotlib.pyplot.bar.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        A matplotlib.figure.Figure instance if return_figures is set to True, otherwise None.
    """

    
    organism1_explained = results['predictivity_summary']['organism1_explained']*100
    organism2_explained = results['predictivity_summary']['organism2_explained']*100
    
    if n_pcs > len(organism1_explained):
        print(str(len(organism1_explained)) + ' PCs at maximum. Set n_pcs to ' + str(len(organism1_explained)))
        n_pcs = len(organism1_explained)

    fig, axs = plt.subplots(1, 2, figsize=(n_pcs, 6))
    
    axs[0].bar([str(i) for i in range(1, n_pcs+1)], organism1_explained[0:n_pcs], color=facecolor[0], **kwargs)
    axs[0].set_title(f'explained {organism1_name} variance ('+ str(np.round(np.sum(organism1_explained[0:n_pcs]),decimals = 2))+'% explained by top '+str(n_pcs)+' PCs)')
    axs[0].set_xlabel(f'{organism1_name}-derived PC')
    axs[0].set_ylabel('Percentage (%)')
    
    axs[1].bar([str(i) for i in range(1, n_pcs+1)], organism2_explained[0:n_pcs], color=facecolor[1], **kwargs)
    axs[1].set_title(f'explained {organism2_name} variance ('+ str(np.round(np.sum(organism2_explained[0:n_pcs]),decimals = 2))+'% explained by top '+str(n_pcs)+' PCs)')
    axs[1].set_xlabel(f'{organism1_name}-derived PC')
    axs[1].set_ylabel('Percentage (%)')
    
    if save_path is not None:
        fig.savefig(save_path + 'var_explained.png', bbox_inches='tight')
    
    if return_figures:
        return fig
    
def gsea_bar(results, GSEA_up_result, GSEA_down_result, n_go=10, facecolor=['#D2493A','#28548f'],save_path=None, return_figures=False, **kwargs):
    
    """
    Visualizes the GSEA (Gene Set Enrichment Analysis) result.
    
    Parameters
    ----------
    GSEA_up_result : dict
        A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for upregulated genes.
    
    GSEA_down_result : dict
        A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for downregulated genes.
    
    n_go : int, optional
        The number of gene ontology terms to consider. Defaults to 10.
    
    facecolor : list, optional
        Colors for the two classes in the plot. Defaults to ['#D2493A','#28548f'].
    
    save_path : str, optional
        Path where the generated figures should be saved. If None, figures won't be saved. Defaults to None.
    
    return_figures : bool, optional
        If True, returns the generated figures. Defaults to False.
    
    kwargs : dict, optional
        Additional keyword arguments to pass to the bar plot function.
    
    Returns
    -------
    list or None
        A list of generated figures if return_figures is set to True. Otherwise, returns None.
    """

    warnings.filterwarnings("ignore")
    figures_up = []
    figures_down = []
    
    top_ids = list(GSEA_up_result.keys())
    
    scores = results['organism2_transComps'].loc[:, top_ids]
    label = results['organism2_classes']
    scores_class1 = scores[label == list(set(label))[0]]
    scores_class2 = scores[label == list(set(label))[1]]

    direction = []
    for i in np.array([int(i) for i in top_ids]):
        if np.mean(scores_class1[i]) > np.mean(scores_class2[i]):
            direction.append('(increases ' + str(list(set(label))[0]) + ' group)')
        elif np.mean(scores_class1[i]) < np.mean(scores_class2[i]):
            direction.append('(decreases ' + str(list(set(label))[0]) + ' group)')
        else:
            direction.append('(groups has no polarization on this PC)')
    
    for i in range(0, len(top_ids)):
        fig, axs = plt.subplots()
        GSEA_up_result[int(top_ids[i])][['-log10_p_adj']][0:n_go].sort_values(by = '-log10_p_adj', ascending = True).plot.barh(xlabel = '-log10 adjusted P-value',color=facecolor[0],legend =None,title = 'GSEA PC'+str(int(top_ids[i])+1)+ direction[i]+ ' Up Pathways',ax =axs, **kwargs)
        figures_up.append(fig)
        
        if save_path is not None:
            fig.savefig(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_UP.png',bbox_inches='tight')
            
    for i in range(0, len(top_ids)):
        fig, axs = plt.subplots()    
        GSEA_down_result[int(top_ids[i])][['-log10_p_adj']][0:n_go].sort_values(by = '-log10_p_adj', ascending = True).plot.barh(xlabel = '-log10 adjusted P-value',color=facecolor[1],legend =None,title = 'GSEA PC'+str(int(top_ids[i])+1)+direction[i] + ' Down Pathways',ax = axs,**kwargs)
        figures_down.append(fig)
        
        if save_path is not None:
            fig.savefig(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_DOWN.png',bbox_inches='tight')

    if return_figures:
            return figures_up, figures_down

def partition(results, organism2_name='organism2', metric='db', top_n_pcs=5,
              pcs=None, covar = None, save_path=None, return_figures=False,
              fontsize=12, fontweight='bold',
              facecolor=["#219EBC", "#FB8500"], cmap ='YlGnBu',  **kwargs):
    """
    Visualizes the Kernel Density Estimation (KDE) of organism2 samples based on the top Principal Components (PCs) 
    that best separate two organism2 classes, as determined by a specified metric.
    
    Parameters
    ----------
    results : dict
        A dictionary from prior computations containing keys 'predictivity_summary' and 'organism2_transComps'.
    
    organism2_name : str, optional
        Name of organism2 for visualization. Defaults to 'organism2'.
    
    metric : str, optional
        Metric for selecting top principal components. Accepts 'db', 'coefs', 'indiv acc', 'ch', 'mean diff', and 'tp'. 
        Defaults to 'db'.
    
    top_n_pcs : int, optional
        The top N principal components under consideration. Defaults to 5.
    
    pcs : int or list of int, optional
        When provided, this specific PC(s) is/are plotted in lieu of the top PCs.
    
    covar : str or list of str, optional
        If provided, the function visualizes the separation of organism2 classes based on the input covariates.
    
    save_path : str, optional
        Directory path where generated figures should be saved. If None, figures are not saved. Defaults to None.
    
    return_figures : bool, optional
        If True, returns the generated figures. Defaults to False.
    
    fontsize : int, optional
        Font size for the plot's labels and title. Defaults to 12.
    
    fontweight : str, optional
        Font weight for the plot's labels and title. Defaults to 'bold'.
    
    facecolor : list, optional
        Colors for the two classes in the plot. Defaults to ["#219EBC", "#FB8500"].
    
    cmap : str or colormap object, optional
        Colormap passed to the heatmap. Utilized if 'covar' contains dummy variables.
    
    kwargs : dict, optional
        Additional keyword arguments for the bar plot function.
    
    Returns
    -------
    list or None
        A list of generated figures if return_figures is True. Otherwise, returns None.
    """
    figures = []
    if covar is not None:
        if not isinstance(covar, list):
            covar = [covar]
        for covar_names in covar:
            scores = results['X_noZ'][covar_names].values
            label = results['y']   # Organism2 classes are considered here
            
            fig, axs = plt.subplots()
            if len(list(set(scores))) ==2:
                df = pd.DataFrame()
                df[covar_names] = scores
                df[organism2_name+' classes'] = np.array(label)
                pivot_table = df.groupby([covar_names, organism2_name+' classes']).size().unstack(fill_value=0)
                sns.heatmap(pivot_table, annot=True, cmap=cmap, cbar_kws={'label': 'Frequency'},ax= axs,**kwargs)
                axs.set_title('Heatmap of classes separation in covariate ' + covar_names, fontsize = 12,fontweight = fontweight)
                figures.append(fig)
                
                if save_path is not None:
                    fig.savefig(save_path+'Partition_covar_'+str(covar_names)+'.png')
            else: 
                scores_class1 = scores[label == list(set(label))[0]]
                scores_class2 = scores[label == list(set(label))[1]]
                scores_plot = np.linspace(np.min(scores), np.max(scores), 100)[:, np.newaxis]
                
                kde1 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class1.reshape(-1, 1))
                log_dens1 = kde1.score_samples(scores_plot)
                axs.fill_between(scores_plot[:,0], np.exp(log_dens1), fc=facecolor[0],**kwargs)
                
                kde2 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class2.reshape(-1, 1))
                log_dens2 = kde2.score_samples(scores_plot)
                axs.fill_between(scores_plot[:,0], np.exp(log_dens2), fc=facecolor[1],**kwargs)
                axs.legend(list(set(label)))
                
                axs.set_title('Organism2 classes separation on covariate '+ covar_names,weight = fontweight)
    
                axs.set_xlabel(covar_names, size=fontsize, weight=fontweight)
                axs.set_ylabel('Normalized Gaussian KDE',size=fontsize, weight=fontweight)
                figures.append(fig)
        
            if save_path is not None:
                fig.savefig(save_path+'Partition_covar_'+str(covar_names)+'.png')
    else:
        if metric == 'db':
            title_kw = 'Davies-Bouldin Score ='
            sort_kw = 'davies_bouldin_score'
            sorted_df = results['predictivity_summary'].sort_values(by=sort_kw, ascending=True)
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                if pcs is not None:
                    if isinstance(pcs, int):
                        top_ids = [pcs]
                    else:
                        top_ids = pcs
            metric_values = sorted_df[sort_kw].loc[top_ids]
            scores = results['organism2_transComps'].iloc[:, top_ids]

        elif metric == 'coefs':
            if sum(results['predictivity_summary']['coefs']!=0)>=top_n_pcs:
                title_kw = 'Regression Coefficient ='
                results['predictivity_summary']['coefs_abs'] = abs(results['predictivity_summary']['coefs'])
                sorted_df = results['predictivity_summary'].sort_values(by = 'coefs_abs',ascending = False)
                
                top_ids = sorted_df.index[0:top_n_pcs]
                if pcs is not None:
                    if pcs is not None:
                        if isinstance(pcs, int):
                            top_ids = [pcs]
                        else:
                            top_ids = pcs
                metric_values = sorted_df['coefs'].loc[top_ids]
                scores = results['organism2_transComps'].iloc[:,top_ids]                   
            else:
                raise ValueError("You don't have enough pcs that have a non-zero regression coef. Decrease your top_n_pcs.")
        elif metric == 'indiv acc':
            title_kw = 'Individual Regression accuracy ='
            sorted_df = results['predictivity_summary'].sort_values(by = 'individual_predictivity',ascending = False) 
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                if pcs is not None:
                    if isinstance(pcs, int):
                        top_ids = [pcs]
                    else:
                        top_ids = pcs
            metric_values = sorted_df['individual_predictivity'].loc[top_ids]
            scores = results['organism2_transComps'].iloc[:,top_ids]
            
        elif metric == 'ch':
            title_kw = 'Calinski-Harabasz Score ='
            sorted_df = results['predictivity_summary'].sort_values(by = 'Calinski-Harabasz Score',ascending = False) 
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                if isinstance(pcs, int):
                    top_ids = [pcs]
                else:
                    top_ids = pcs
            metric_values = sorted_df['Calinski-Harabasz Score'].loc[top_ids]
            scores = results['organism2_transComps'].iloc[:,top_ids]
            
        elif metric == 'mean diff':
            title_kw = 'Difference of Means ='
            sorted_df = results['predictivity_summary'].sort_values(by = 'Difference of Means',ascending = False) 
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                if pcs is not None:
                    if isinstance(pcs, int):
                        top_ids = [pcs]
                    else:
                        top_ids = pcs
            metric_values = sorted_df['Difference of Means'].loc[top_ids]
            scores = results['organism2_transComps'].iloc[:,top_ids]
            
        elif metric == 'tp':
            title_kw = 't-test p-value ='
            sorted_df = results['predictivity_summary'].sort_values(by = 't-test p-value',ascending = True) 
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                if pcs is not None:
                    if isinstance(pcs, int):
                        top_ids = [pcs]
                    else:
                        top_ids = pcs
            metric_values = sorted_df['t-test p-value'].loc[top_ids]
            scores = results['organism2_transComps'].iloc[:,top_ids]

        else:
            raise ValueError("Invalid metric. Please choose one from 'coefs', 'indiv acc', 'db','ch','mean diff','tp'.")

        label = results['organism2_classes']   # Organism2 classes are considered here
        scores_class1 = scores[label == list(set(label))[0]]
        scores_class2 = scores[label == list(set(label))[1]]
        
        for i in range(0, len(top_ids)):
            fig, axs = plt.subplots()
            scores_plot = np.linspace(np.min(scores.iloc[:,i]), np.max(scores.iloc[:,i]), 100)[:, np.newaxis]
            
            kde1 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class1.iloc[:,i].values.reshape(-1, 1))
            log_dens1 = kde1.score_samples(scores_plot)
            axs.fill_between(scores_plot[:,0], np.exp(log_dens1), fc=facecolor[0],**kwargs)
            
            kde2 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class2.iloc[:,i].values.reshape(-1, 1))
            log_dens2 = kde2.score_samples(scores_plot)
            axs.fill_between(scores_plot[:,0], np.exp(log_dens2), fc=facecolor[1],**kwargs)
            axs.legend(list(set(label)))
            
            axs.set_title(title_kw + '%s' % float('%.3g' % metric_values.iloc[i]),weight = fontweight )

            axs.set_xlabel(organism2_name+' TransComp ' + str(scores.columns[i]+1) + ' Scores', size=fontsize, weight=fontweight)
            axs.set_ylabel('Normalized Gaussian KDE',size=fontsize, weight=fontweight)
            figures.append(fig)
        
            if save_path is not None:
                fig.savefig(save_path+'Partition_TransComp'+str(int(str(top_ids[i]))+1)+'_'+metric+'.png')


    if return_figures:
        return figures

def covariate_pc_interaction(results, organism2_covar, covar_key, pc, is_categorical = True,
                             nbins = 5, facecolor=["#219EBC", "#FB8500"],
                             save_path = None,return_figures=False,**kwargs):
    #single pair of covar_pcs
    #Again, python index starts from 0, if you want pc1, type 0 here
    """
    Visualizes the interaction between covariates and principal components (PCs) using either a box plot for categorical covariates or a bar plot for continuous covariates.
    
    Parameters
    ----------
    results : dict
        Dictionary with information about organisms and their associated PCs.
        - 'organism2_transComps': DataFrame with organisms as indices and PCs as columns.
        - 'organism2_classes': DataFrame or column indicating the class/category of each organism.
    
    organism2_covar : dict
        Dictionary containing covariate information for each organism.
    
    covar_key : str
        Key to access specific covariate information within the organism2_covar dictionary.
    
    pc : int
        Index of the principal component (0-based index, e.g., 0 for PC1).
    
    is_categorical : bool, optional
        If the covariate is categorical or not. Defaults to True. If False, assumes continuous.
    
    nbins : int, optional
        Number of bins for binning continuous covariates. Used only if is_categorical is False. Defaults to 5.
    
    facecolor : list of str, optional
        List of two colors for plotting. First color is for class1 and second for class2. Defaults to ["#219EBC", "#FB8500"].
    
    save_path : str, optional
        If provided, saves the resulting figure to this path. Defaults to None.
    
    return_figures : bool, optional
        If True, returns the generated figure object. Otherwise, displays it. Defaults to False.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figures is True, returns the generated figure. Otherwise, None.
    
    Examples
    --------
    To visualize the interaction of a categorical covariate with PC1:
    >>> covariate_pc_interaction(results_dict, organism_covar_dict, 'covariate_key', 0)
    
    To visualize the interaction of a continuous covariate with PC1 and save the figure:
    >>> covariate_pc_interaction(results_dict, organism_covar_dict, 'covariate_key', 0, is_categorical=False, save_path='./path/to/save/')
    """

   
    if is_categorical:
        df = pd.DataFrame()
        df.index = results['organism2_transComps'].index
        df['score'] = results['organism2_transComps'].iloc[:,pc].values
        df['covar_cat'] = organism2_covar[covar_key].astype('category')
        df['classes'] = results['organism2_classes']
        df['covar_cat_classes'] = df['covar_cat'].astype(str).map(str) + '_'+df['classes'].astype(str).map(str)
        df = df.sort_values(by = 'covar_cat_classes')
        fig, axs = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='covar_cat_classes', y='score', data=df, ax = axs,**kwargs)
        for label in axs.get_xticklabels():
            label.set_rotation(90)
        axs.set_ylabel('PC '+ str(pc+1)+' Score')
        axs.set_xlabel(covar_key+' + classes')
        if save_path is not None:
            fig.savefig(save_path+organism2_covar+'_PC'+str(pc+1)+'.png',bbox_inches='tight')
        if return_figures:
            return fig
    else:
        print('Assume the covariate is continuous')
        df = pd.DataFrame()
        df.index = results['organism2_transComps'].index
        df['score'] = results['organism2_transComps'].iloc[:,pc].values
        df['classes'] = results['organism2_classes']
        
        continuous_covar = []
        for i in organism2_covar[covar_key]:
            try:
                continuous_covar.append(float(i))
            except:
                continuous_covar.append(np.nan)
        data = np.array(continuous_covar)
        df['continuous_covar'] = data
        df=df.dropna()
        
        bin_edges = np.floor(np.linspace(start=np.nanmin(df['continuous_covar']), stop=np.nanmax(df['continuous_covar']), num=nbins+1))
        # Create bin labels with ranges
        bin_labels = [f"{bin_edges[i]} to {bin_edges[i+1]}" for i in range(nbins)]
        # Use pandas.cut to bin the data
        binned_data = pd.cut(df['continuous_covar'], bins=bin_edges, labels=bin_labels, include_lowest=True)
        # Convert binned_data to an array of strings (bin names)
        group_names = np.array(binned_data.astype(str))
        
        df['covar_cat'] = group_names
        class1_avg_scores = df.loc[df['classes'] == list(set(df['classes']))[0],:].groupby('covar_cat')['score'].mean()
        class2_avg_scores = df.loc[df['classes'] == list(set(df['classes']))[1],:].groupby('covar_cat')['score'].mean()

        fig, axs = plt.subplots(figsize=(6, 6))
        # The width of the bars
        bar_width = 0.5  # bars will now take up half of their designated space
        
        # Plot bars for each class
        axs.bar(class1_avg_scores.index, class1_avg_scores, color=facecolor[0], width=bar_width, edgecolor='black', alpha=0.5, label=list(set(df['classes']))[0], align='edge',**kwargs)
        axs.bar(class2_avg_scores.index, class2_avg_scores, color=facecolor[1], width=bar_width, edgecolor='black', alpha=0.5, label=list(set(df['classes']))[1], align='center',**kwargs)
        
        # Setting the labels, title, and custom x-axis tick labels
        axs.set_ylabel('PC '+ str(pc+1)+' Score')
        axs.set_xlabel(covar_key+' groups')
        axs.set_xticklabels(bin_labels, rotation=90)
        axs.legend()  # display the legend
        
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path+organism2_covar+'_PC'+str(pc+1)+'.png',bbox_inches='tight')
        if return_figures:
            return fig