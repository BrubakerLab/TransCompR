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
import gseapy as gp
from sklearn.neighbors import KernelDensity

def loadings(results, pcs, top_n_genes = 20, save_path=None, return_figures=False, 
             figsize=(8,8), fontsize=20, fontweight='bold',
             textsize=8,textcolor='red', textweight='bold',
             stripdotsize=3, highlightcolor='orange', jitter=0.3, 
             scattersize = 5):
    
    """
    Visualize loadings of mouse genes on each mouse principal components.
    
    Args:
    results (dict): Results dictionary with key 'mouse_loadings' containing a DataFrame of mouse gene loadings.
    pcs (list or int): A list or an integer indicating which principal components to plot.
    top_n_genes (int): Positive integer indicate how many top genes should be highlighted.
    save_path (str, optional): Path to save the figures. If None, figures will not be saved. Default to None.
    return_figures (bool, optional): Whether to return the matplotlib.figure.Figure instances. If False, nothing will be returned. Default to False.
    figsize (tuple, optional): Size of the figure. Default to (8,8).
    fontsize (int, optional): Font size for plot labels and title. Default to 30.
    fontweight (str, optional): Font weight for plot labels and title. Default to 'bold'.
    textsize (int, optional): Text size for annotations. Default to 8.
    textcolor (str, optional): Text color for annotations. Default to 'red'.
    textweight (str, optional): Text weight for annotations. Default to 'bold'.
    stripdotsize (int, optional): Size of the dots in the strip plot. Default to 3.
    highlightcolor (str, optional): Color to highlight the outliers. Default to 'orange'.
    jitter (float, optional): Amount of jitter (only along the categorical axis) to apply. This can be useful when you have many points and they overlap, so that it is easier to see the distribution. Default to 0.3.
    scattersize (int, optional): Size of the dots in the scatter plot. Default to 5.
    
    Returns:
    figures (list, optional): List of matplotlib.figure.Figure instances if return_figures is True, else None.
    
    Raises:
    ValueError: If pcs is not an integer or a list of 2 integers.
    """

    mouse_loadings = results['mouse_loadings']
    
    figures = []
    
    for i in pcs:
        if isinstance(i,int):
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data = mouse_loadings.iloc[i,:].T.sort_values()
           
            outliers =  pd.concat([data.head(int(np.ceil(top_n_genes/2))),data.tail(int(np.floor(top_n_genes/2)))])
            insiders = data[int(np.ceil(top_n_genes/2)):-int(np.floor(top_n_genes/2))]

            path_collection = sns.stripplot(data=outliers,ax=ax, color=highlightcolor, size = stripdotsize*2, jitter=jitter).collections[0]
            coords = path_collection.get_offsets()
            xcoord= [coords[i][0] for i in range(0,len(outliers))]
            ycoord= [coords[i][1] for i in range(0,len(outliers))]

            texts = []
            for j,txt in enumerate(outliers.index):
                texts.append(ax.text(xcoord[j], ycoord[j], txt,size=textsize, color=textcolor, weight=textweight))
        
            sns.boxplot(data=data,ax=ax, fliersize=0, width=jitter*2)
            sns.stripplot(data=insiders,ax=ax, color=(0.5, 0.5, 0.5, 0.6), size = stripdotsize, jitter=jitter,zorder= 1)

            ax.set_xlabel('Mouse PC '+ str(i+1),size=fontsize, weight=fontweight)
            ax.set_ylabel('Loadings',size=fontsize, weight=fontweight)
            ax.set_xticks([0])
            ax.set_xticklabels('')
            adjust_text(texts)
            figures.append(fig)
            
        elif isinstance(i, list) and all([isinstance(j,int) for j in i]):
            if len(i) ==2:
                data = mouse_loadings.iloc[i,:].T
                data['abs_sum'] = abs(data[i[0]])+abs(data[i[1]])
                plot_entries =  data.sort_values(by='abs_sum',ascending=False).head(top_n_genes)
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
                
                ax.scatter(data.iloc[:,0],data.iloc[:,1], s = scattersize, color = 'grey')
                ax.scatter(plot_entries.iloc[:,1],plot_entries.iloc[:,2], s = scattersize*2, color =highlightcolor )
    
                texts = []
                for j,txt in enumerate(labels):
                    texts.append(ax.text(plot_entries.iloc[j,1], plot_entries.iloc[j,2], txt, size=textsize, color=textcolor, weight=textweight))
                adjust_text(texts)
                
                ax.text(0, 0.5, 'PC'+str(i[0]+1), va='center', ha='right', size=fontsize, transform=ax.transAxes, weight=fontweight)
                ax.text(0.5, 0, 'PC'+str(i[1]+1), va='top', ha='center', size=fontsize, transform=ax.transAxes, weight=fontweight)
                figures.append(fig)
            else:
                raise ValueError("Only support 2D plot.")
        else:
            raise ValueError("Invalid pcs argument.")
    
        if save_path is not None:
            fig.savefig(save_path+'PC'+str(np.array(i)+1)+'.png')

    if return_figures:
        return figures
    
def var_explained(results, n_pcs = 20, save = None,return_figures=False, plot_kwargs1=None, plot_kwargs2=None):
    
    """
    Visualize the total variance explained in mouse and human data by mouse principal components (PCs).
    
    Args:
    results (dict): Results dictionary with key 'predictivity_summary' containing a sub-dictionary with keys 'human_explained' and 'mouse_explained' referring to the percentage variance explained by mouse PCs in human and mouse data, respectively.
    n_pcs (int, optional): Number of PCs to be included in the visualization. Default to 20.
    save (str, optional): Path to save the figure. If None, the figure will not be saved. Default to None.
    return_figures (bool, optional): Whether to return the matplotlib.figure.Figure instance. If False, nothing will be returned. Default to False.
    plot_kwargs1 (dict, optional): Additional keyword arguments to pass to matplotlib.pyplot.bar for the mouse variance plot. Default to None.
    plot_kwargs2 (dict, optional): Additional keyword arguments to pass to matplotlib.pyplot.bar for the human variance plot. Default to None.
    
    Returns:
    fig (matplotlib.figure.Figure, optional): A matplotlib.figure.Figure instance if return_figures is True, else None.
    """
    
    human_explained = results['predictivity_summary']['human_explained']
    mouse_explained = results['predictivity_summary']['mouse_explained']
    
    if plot_kwargs1 is None:
        plot_kwargs1 = {}
    if plot_kwargs2 is None:
        plot_kwargs2 = {}
    if n_pcs> len(mouse_explained):
        print(str(len(mouse_explained))+' PCs at maximum. Set n_pcs to '+str(len(mouse_explained)))
        n_pcs = len(mouse_explained)

    fig, axs = plt.subplots(1, 2, figsize=(n_pcs, 6))
    
    axs[0].bar([str(i) for i in range(1,n_pcs+1)], mouse_explained[0:n_pcs], color='#00AFB9', **plot_kwargs1)
    axs[0].set_title('explained mouse variance')
    axs[0].set_xlabel('Mouse PC')
    axs[0].set_ylabel('Percentage')
    
    axs[1].bar([str(i) for i in range(1,n_pcs+1)], human_explained[0:n_pcs], color='#F07167', **plot_kwargs2)
    axs[1].set_title('explained human variance')
    axs[1].set_xlabel('Mouse PC')
    axs[1].set_ylabel('Percentage')
    
    if save is not None:
        plt.savefig(save,bbox_inches='tight')
    
    if return_figures:
        return fig
    
def gsea(results,  metric = 'db', top_n_pcs = 5, pcs = None,top_n_genes = 100, n_go = 10, save_path = None, return_figures=False,**kwargs):
    #metrics: 'coefs', 'indiv acc', 'db'
    """
    This function performs Gene Set Enrichment Analysis (GSEA) on genes that have the largest loadings 
    on the principal components (PCs) that best separate two human classes. 
    The PCs are selected according to three possible metrics: 'db' (Davies Bouldin Score), 'coefs' (Coefficients), 
    or 'indiv acc' (Individual Accuracy).
    
    Args:
        results (dict): A dictionary containing the results of previous computations. 
                        Expected to contain the key 'predictivity_summary'.
        metric (str): The metric to be used for selecting top principal components. 
                      Accepts 'db', 'coefs', 'indiv acc'. Default is 'db'.
        top_n_pcs (int): The top N principal components to be considered. Default is 5.
        pcs (int or list of int): When specified, the specific pc is plotted instead of top pcs.
        top_n_genes (int): The top N genes to be considered for each principal component. Default is 100.
        n_go (int): The number of gene ontology terms to consider. Default is 10.
        save_path (str): The directory path where generated figures should be saved. If None, figures will not be saved.
        return_figures (bool): If True, the figures generated by the function will be returned. Default is False.
        kwargs (dict): Additional keyword arguments to pass to the bar plot function.
    
    Returns:
        tuple: A tuple containing the following items:
            - figures_up (list): A list of figures representing GSEA for genes upregulated in the top PCs.
            - figures_down (list): A list of figures representing GSEA for genes downregulated in the top PCs.
            - GESA_up_result (dict): A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for upregulated genes.
            - GESA_down_result (dict): A dictionary where keys are the top PCs and values are DataFrames of the GSEA results for downregulated genes.
    """
    figures_up = []
    figures_down = []
    if metric == 'db':
        sorted_df = results['predictivity_summary'].sort_values(by = 'davies_bouldin_score',ascending = True)
    elif metric == 'coefs':
        if sum(results['predictivity_summary']['coefs']!=0)>=top_n_pcs:
            results['predictivity_summary']['coefs_abs'] = abs(results['predictivity_summary']['coefs'])
            sorted_df = results['predictivity_summary'].sort_values(by = 'coefs_abs',ascending = False)
        else:
            raise ValueError("You don't have enough pcs that have a non-zero coef. Decrease your top_n_pcs.")
    elif metric == 'indiv acc':
        sorted_df = results['predictivity_summary'].sort_values(by = 'individual_predictivity',ascending = False)  
    else:
        raise ValueError("Invalid metric. Please choose one from 'coefs', 'indiv acc', 'db'.")
    
    top_ids = sorted_df.index[0:top_n_pcs]
    if pcs is not None:
        top_ids = pcs
    top_pc_genes = sorted_df['sorted_mouse_genes'].loc[top_ids]
    down_genes = [i[0:top_n_genes] for i in top_pc_genes]
    up_genes = [i[(0-top_n_genes):] for i in top_pc_genes]
    
    scores = results['human_transComps'].loc[:,top_ids]
    label = results['human_classes']
    scores_class1 = scores[label == list(set(label))[0]]
    scores_class2 = scores[label == list(set(label))[1]]
    
    direction = []
    for i in np.array([int(i) for i in top_ids]):
        if np.mean(scores_class1[i]) > np.mean(scores_class2[i]):
            direction.append('(increases '+str(list(set(label))[0])+ ' group)')
        elif np.mean(scores_class1[i]) < np.mean(scores_class2[i]):
            direction.append('(decreases '+ str(list(set(label))[0]) + ' group)')
        else:
            direction.append('(groups has no polarization on this PC)')

    GESA_up_result = dict()
    for i in range(0, len(top_ids)):
        fig, axs = plt.subplots()
            
        GOBP_up = gp.enrichr(gene_list=up_genes[i] ,
                             gene_sets=['GO_Biological_Process_2021'],
                             organism='Mouse', 
                             outdir='test/enr_DEGs_GOBP_ec',
                             )
        GESA_up_result[int(top_ids[i])] = GOBP_up.results
        
        GOBP_up.results['-log10_p_adj'] = -np.log10(GOBP_up.results['Adjusted P-value'])
        GOBP_up.results.index = GOBP_up.results['Term']
        GOBP_up.results[['-log10_p_adj']][0:n_go].sort_values(by = '-log10_p_adj', ascending = True).plot.barh(xlabel = '-log10 adjusted P-value',legend =None,title = 'GSEA PC'+str(int(top_ids[i])+1)+' UP '+ direction[i],ax =axs, **kwargs)
        figures_up.append(fig)
        if save_path is not None:
            fig.savefig(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_UP.png',bbox_inches='tight')

    GESA_down_result = dict()
    for i in range(0, len(top_ids)):
        fig, axs = plt.subplots()    
        GOBP_down = gp.enrichr(gene_list=down_genes[i] ,
                             gene_sets=['GO_Biological_Process_2021'],
                             organism='Mouse', 
                             outdir='test/enr_DEGs_GOBP_ec',
                             )
        GESA_down_result[int(top_ids[i])] = GOBP_down.results
        
        GOBP_down.results['-log10_p_adj'] = -np.log10(GOBP_down.results['Adjusted P-value'])
        GOBP_down.results.index = GOBP_down.results['Term']
        GOBP_down.results[['-log10_p_adj']][0:n_go].sort_values(by = '-log10_p_adj', ascending = True).plot.barh(xlabel = '-log10 adjusted P-value',legend =None,title = 'GSEA PC'+str(int(top_ids[i])+1)+' DOWN '+ direction[i],ax = axs,**kwargs)
        figures_down.append(fig)
        
        if save_path is not None:
            fig.savefig(save_path+'GSEA_PC'+str(int(top_ids[i])+1)+'_DOWN.png',bbox_inches='tight')
            
    if return_figures:
        return GESA_up_result, GESA_down_result,figures_up, figures_down
    else:
        return GESA_up_result, GESA_down_result
    

def partition(results,  metric = 'db', top_n_pcs = 5,pcs = None, save_path = None, return_figures=False,
              fontsize = 12, fontweight= 'bold', facecolor = ["blue","orange"],  **kwargs):
    #metrics: 'coefs', 'indiv acc', 'db'
    """
    This function visualizes the Kernel Density Estimation (KDE) of human samples based on top Principal Components (PCs) 
    that can best separate two human classes, according to a specified metric.
    
    Args:
        results (dict): A dictionary containing the results of previous computations. 
                        Expected to contain the key 'predictivity_summary' and 'human_transComps'.
        metric (str): The metric to be used for selecting top principal components. 
                      Accepts 'db', 'coefs', 'indiv acc'. Default is 'db'.
        pcs (int or list of int): When specified, the specific pc is plotted instead of top pcs.
        top_n_pcs (int): The top N principal components to be considered. Default is 5.
        save_path (str): The directory path where generated figures should be saved. If None, figures will not be saved.
        return_figures (bool): If True, the figures generated by the function will be returned. Default is False.
        fontsize (int): Font size for the plot title and labels. Default is 12.
        fontweight (str): Font weight for the plot title and labels. Default is 'bold'.
        facecolor (list): List of colors to be used for the two classes in the plot. Default is ["blue", "orange"].
        kwargs (dict): Additional keyword arguments to pass to the bar plot function.
    
    Returns:
        list: A list of figures if return_figures is set to True. Otherwise, it doesn't return anything.
    """
    figures = []
    if metric == 'db':
        title_kw = 'Davies-Bouldin Score ='
        sort_kw = 'davies_bouldin_score'
        sorted_df = results['predictivity_summary'].sort_values(by = 'davies_bouldin_score',ascending = True)
        top_ids = sorted_df.index[0:top_n_pcs]
        if pcs is not None:
            top_ids = pcs
        metric_values = sorted_df['davies_bouldin_score'].loc[top_ids]
        scores = results['human_transComps'].iloc[:,top_ids]
        
    elif metric == 'coefs':
        if sum(results['predictivity_summary']['coefs']!=0)>=top_n_pcs:
            title_kw = 'Regression Coefficient ='
            results['predictivity_summary']['coefs_abs'] = abs(results['predictivity_summary']['coefs'])
            sorted_df = results['predictivity_summary'].sort_values(by = 'coefs_abs',ascending = False)
            
            top_ids = sorted_df.index[0:top_n_pcs]
            if pcs is not None:
                top_ids = pcs
            metric_values = sorted_df['coefs'].loc[top_ids]
            scores = results['human_transComps'].iloc[:,top_ids]       
            
        else:
            raise ValueError("You don't have enough pcs that have a non-zero regression coef. Decrease your top_n_pcs.")
    elif metric == 'indiv acc':
        title_kw = 'Individual Regression accuracy ='
        sorted_df = results['predictivity_summary'].sort_values(by = 'individual_predictivity',ascending = False) 
        top_ids = sorted_df.index[0:top_n_pcs]
        if pcs is not None:
            top_ids = pcs
        metric_values = sorted_df['individual_predictivity'].loc[top_ids]
        scores = results['human_transComps'].iloc[:,top_ids]

    else:
        raise ValueError("Invalid metric. Please choose one from 'coefs', 'indiv acc', 'db'.")

    label = results['human_classes']
    scores_class1 = scores[label == list(set(label))[0]]
    scores_class2 = scores[label == list(set(label))[1]]
    
    for i in range(0, len(top_ids)):
        fig, axs = plt.subplots()
        scores_plot = np.linspace(np.min(scores.iloc[:,i]), np.max(scores.iloc[:,i]), 100)[:, np.newaxis]
        
        kde1 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class1.iloc[:,i].values.reshape(-1, 1))
        log_dens1 = kde1.score_samples(scores_plot)
        axs.fill_between(scores_plot[:,0], np.exp(log_dens1), fc=facecolor[0])
        
        kde2 = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(scores_class2.iloc[:,i].values.reshape(-1, 1))
        log_dens2 = kde2.score_samples(scores_plot)
        axs.fill_between(scores_plot[:,0], np.exp(log_dens2), fc=facecolor[1])
        axs.legend(list(set(label)))
        
        axs.set_title(title_kw + '%s' % float('%.3g' % metric_values.iloc[i]),weight = fontweight )

        axs.set_xlabel('Mouse PC '+ str(scores.columns[i]+1) + ' Scores',size=fontsize, weight=fontweight)
        axs.set_ylabel('Normalized Gaussian KDE',size=fontsize, weight=fontweight)
        figures.append(fig)
    
        if save_path is not None:
            fig.savefig(save_path+'PC'+str(top_ids[i])+'.png')

    if return_figures:
        return figures
