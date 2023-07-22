# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:51:02 2023

@author: Ran
"""

import time
from Bio import Entrez
import pandas as pd
import numpy as np
from .utils import  indices

def homolo_rough_align(human_df, mice_df):
    """
    Roughly aligns human and mouse homologous genes based on capitalization.
    It's a quick approach that is very biased and may result in errors due to inconsistent capitalization.
    
    Args:
        human_df (pandas.DataFrame): The DataFrame containing human gene expression data. Columns should be gene names.
        mice_df (pandas.DataFrame): The DataFrame containing mice gene expression data. Columns should be gene names.
    
    Returns:
        tuple: A tuple containing two DataFrames (human_df, mice_df) with reordered columns to match homologous genes.
    """
    print('Warning: Given the sever of ensembl is somewhat unstable, this is a very biased way of quickly matching human and mouse homologs based on the capitalization.')
    human_genes = human_df.columns.tolist()
    
    lower_human_genes = [i[0]+i[1:].lower() for i in human_genes]
    intersection = mice_df.columns[mice_df.columns.isin(lower_human_genes)]
    keep_genes = [(i in intersection) for i in lower_human_genes]
    order = [lower_human_genes[i] for i in indices(keep_genes,True)]
    
    mice_df= mice_df[order]
    human_df = human_df.reindex(columns = [i[0]+i[1:].upper() for i in order])
    return human_df, mice_df

def homolo_align(human_df,mouse_df, email):
    print('Warning: Given the sever of ensembl is somewhat unstable, this is a slow way of matching human and mouse homologs by inquiring NCBI database. Some mouse genes are not directly linked to their human homologs in NCBI, so use this function with cautious. The speed is about 1 gene per 3 seconds.')
    # Always tell NCBI who you are
    Entrez.email = email
    """
    Accurately aligns human and mouse homologous genes by querying the NCBI database.
    Some mouse genes are not directly linked to their human homologs in NCBI, so this function may miss some genes.

    Args:
        human_df (pandas.DataFrame): The DataFrame containing human gene expression data. Columns should be gene names.
        mouse_df (pandas.DataFrame): The DataFrame containing mice gene expression data. Columns should be gene names.
        email (str): The email address to be used with the NCBI Entrez utility.

    Returns:
        tuple: A tuple containing two DataFrames (human_df, mouse_df) with reordered columns to match homologous genes.
    """

    print('Warning: Given the server instability, this is a slow way of matching human and mouse homologs by inquiring NCBI database. Some mouse genes are not directly linked to their human homologs in NCBI, so use this function with caution. The speed is about 1 gene per 3 seconds.')
    
    def indices(lst, item):
    
    def find_homologs(human_genes):
        def get_mouse_homologs(human_gene_name):
            search_result = Entrez.esearch(db="gene", term=f"{human_gene_name}[Gene Name] AND Homo sapiens[Orgn]")
            record = Entrez.read(search_result)
    
            # If there are results, get the gene id
            if record["Count"] != "0":
                gene_id = record["IdList"][0]
    
                # Fetch the gene
                gene = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
                gene_record = Entrez.read(gene)
    
                # Find the HomoloGene ID and use it to fetch the homologs
                homologene_id = None
                for item in gene_record[0]["Entrezgene_comments"]:
                    if 'Gene-commentary_heading' in item.keys():
                        if item['Gene-commentary_heading']=='Orthologs from Annotation Pipeline':
                            for comments in item['Gene-commentary_comment']:
                                for sources in comments['Gene-commentary_source']:
                                    if 'Other-source_anchor' in sources.keys():
                                        if sources['Other-source_anchor'] == 'mouse':
                                            homologene_id = sources['Other-source_src']['Dbtag']['Dbtag_tag']['Object-id']['Object-id_id'] 
                                            break
                                        
                if homologene_id is not None:
                    homologs = Entrez.efetch(db="gene", id=homologene_id, retmode="xml")
                    mouse_homologs_record = Entrez.read(homologs)
                    mouse_homologs_name = str(mouse_homologs_record[0]['Entrezgene_gene']['Gene-ref']['Gene-ref_locus'])
                else:
                    mouse_homologs_name = 'None'
                return mouse_homologs_name
         
        delay = 1
        homologs_dict = {}
        for gene_name in human_genes:
            print(gene_name)
            while True:
                try:
                    mouse_homologs_name = get_mouse_homologs(gene_name)
                    print(mouse_homologs_name)
                    homologs_dict[gene_name] =mouse_homologs_name
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(f"Waiting for {delay} seconds before trying again.")
                    time.sleep(delay)  # Wait for the specified delay before trying again
        return homologs_dict
    
    human_genes = human_df.columns.tolist()
    homologs_dict = find_homologs(human_genes)
    homologs_df = pd.DataFrame.from_dict(homologs_dict, orient= 'index')
    drop_genes = [human_genes[i] for i in indices(homologs_df.transpose().values.tolist()[0],'None')]
    human_df = human_df.drop(drop_genes, axis = 1)
    
    corres_mouse_genes = homologs_df.transpose()[human_df.columns].values.tolist()[0]
    mouse_df = mouse_df.iloc[:,mouse_df.columns.isin(corres_mouse_genes)]
    ordered_human_genes = [homologs_df[homologs_df[0] == i].index.tolist()[0] for i in mouse_df.columns]
    human_df = human_df[ordered_human_genes]

    return human_df, mouse_df


def highly_variable_genes(df, n_top_genes):
    """
    Identifies the most variable genes in the dataset.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing gene expression data. Columns should be gene names and rows should correspond to individual samples.
        n_top_genes (int): The number of top variable genes to return.
        
    Returns:
        list: A list of the most variable genes.
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
    
    # Selecting the most variable genes
    df_fluctuation = df_fluctuation.sort_values(by='dispersion', ascending=False)
    hvg = df_fluctuation.head(n_top_genes).index.tolist()
    
    return hvg