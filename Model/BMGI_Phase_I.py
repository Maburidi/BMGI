"""
This script implements phase I in BMGI algorithm: more detials can be found in BMGI paper: 

Random Forest is used here to score every gene for classification power using its \
isoforms expression values as features.

Required Inputs:
1- Annotated gene sets. Each gene is represented by an Ensemble ID (e.g. ). txt file
2- TEM: sigle-cell processed transcriptomic expression data (rows are the samples (i.g. cells), columns are the transcripts). csv file
3- GEM: sigle-cell processed gene expression data (rows are the samples (i.g. cells), columns are the genes). csv file
4- Sample labels (samples (i.g. cells) belongs to classes, each samples should have a label). csv file

Not required inputs:
5- P-value threshold (defult = 0.001).
6- Percentage threshold (defult = 0.6). 
7- Number of trees for Random Forest (defult =100). 
8- Number of folds for cross validation (defult =10). 

Outputs:
1- 



# Example:

python Model/BMGI_Phase_I.py \
    --gene_sets    /home/maburid/AMIA_project/msigdb.v6.2.symbols.txt \
    --TEM_dataset  /home/maburid/AMIA_project/Transc_stem_cell.csv \
    --GEM_dataset  /home/maburid/AMIA_project/Gene_stem_cell.csv \
    --labels       /home/maburid/AMIA_project/labels.csv \
    --num_trees    50 \
    --cv           5

"""


import gc
gc.collect()
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.mixture
import scipy
import argparse

import random
import sys
import json
import time
import mygene
import gseapy
from gseapy import parser 
import pandas as pd                        
import xlrd        
import os
from sklearn import preprocessing
import operator

def readENS_ids(path_to_ref):        
    TX_to_ENSG={}                   
    ENSG_isoforms={}                    
    line_cnt=0;                           
    tx_cnt=0;                     
    i=0;                              
    with open(path_to_ref) as f:                                                                      
        for line in f:                               
            if line[0]=='>':                                                                      
                s=line.split('ENST',1)[1][:11]                                                           
                if int(s) in TX_SET:                                                                       
                    if tx_cnt != 172941 and tx_cnt != 172942 :                                                                                  
                        TX_to_ENSG['ENST'+s] = 'ENSG'+line.split('ENSG',1)[1][:11]                                                                                             
                        ENSG_isoforms[TX_to_ENSG['ENST'+s]] = ENSG_isoforms.get(TX_to_ENSG['ENST'+s], [])  
                        ENSG_isoforms[TX_to_ENSG['ENST'+s]].append('ENST'+s)                                            
                    tx_cnt+=1                                                                               
                    if tx_cnt==len(TX_SET): return [TX_to_ENSG,ENSG_isoforms]                                 
                
                line_cnt+=1
    return [TX_to_ENSG,ENSG_isoforms]

def symb_to_id(Id_set):                                                                   
    symboles_set =[]                                                                    
    mg = mygene.MyGeneInfo()                                                          
    ginfo = mg.querymany(Id_set, scopes='symbol',fields=["ensembl"],species="human")  
    for i in range(len(Id_set)):                                                      
        if 'notfound' in ginfo[i]:                                                    
            continue                                                                   
        if "ensembl" in list(ginfo[i].keys()):                                         
            if type(ginfo[i]["ensembl"]) is list:                                      
                for j in range(len(ginfo[i]["ensembl"])):                               
                    symboles_set.append(ginfo[i]["ensembl"][j]["gene"])                               
            else:                                                                          
                symboles_set.append(ginfo[i]["ensembl"]["gene"])                         
    return symboles_set                                                                                                                                 



if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Select genes which perform significantly better than an equivalent random gene for each annotated set.")
    parser.add_argument("--gene_sets", help="list of annotated gene sets", required=True)
    parser.add_argument("--TEM_dataset", help="TEM input dataset (samples x transcripts)", required=True)
    parser.add_argument("--GEM_dataset", help="GEM input dataset (samples x genes)", required=True)
    parser.add_argument("--labels", help="list of sample labels", required=True)
    
    parser.add_argument("--num_trees", help="number of trees in random forest", type=int, default=100)
    parser.add_argument("--cv", help="number of folds for k-fold cross validation", type=int, default=10)
    parser.add_argument("--p_threshold", help="maximum p-value required for a gene to be selected", type=float, default=0.001)
    parser.add_argument("--percent_threshold", help="minimum percentage of filtered genes in a set to the origional genes, otherwise, if less \
    than this threshould, dont count the set as a significant", type=float, default=0.6)
    
    args = parser.parse_args()

    # load input data
    print("loading input datasets...")
    filename_gene_sets= args.gene_sets
    filename_TEM = args.TEM_dataset
    filename_GEM = args.GEM_dataset 
    filename_labels = args.labels
    
    Trans_data=pd.read_csv(filename_TEM)                             
    Trans_data= Trans_data.set_index("Unnamed: 0")        #
    Gene_data=pd.read_csv(filename_GEM)        
    Gene_data= Gene_data.set_index("Unnamed: 0")         #
    labels = pd.read_csv(filename_labels) 
    
    # load annotated gene sets
    with open(filename_gene_sets) as f:
    gene_sets = []
    for line in f:
        line = line.split()  
        gene_sets.append(line)
    
    
    #=======================================
    # not needed
    gene_names = Gene_data.index
    ENSG_idx= np.array([gene_names[i] for i in range(len(gene_names)) if gene_names[i][:4]=='ENSG'])
    Gene_data = Gene_data.transpose()
    Gene_data = Gene_data[ENSG_idx]
    gene_names= Gene_data.columns
    gene_names=np.array([t.split('.')[0] for t in gene_names])  
    Gene_data.columns = gene_names

    Trans_names = Trans_data.index
    ENST_idx= np.array([Trans_names[i] for i in range(len(Trans_names)) if Trans_names[i][:4]=='ENST'])
    Trans_data = Trans_data.transpose()
    Trans_data = Trans_data[ENST_idx]
    Trans_names= Trans_data.columns
    Trans_names=np.array([t.split('.')[0] for t in Trans_names])  
    Trans_data.columns = Trans_names

    #=======================================
    # needed 
    print("Genes_level_data_shape:  ", Gene_data.shape )
    print("Genes_level_data_shape:  ", Trans_data.shape )
    print("Number of Annotated sets: ", len(gene_sets))
    print("label_length:  ", len(labels) )
    
    #=======================================================================================================
    # not needed
    classes = ["D0_H7_hESC","D1_H7_derived_APS","D1_H7_derived_MPS","D2_H7_derived_DLL1pPXM", "D3_H7_derived_ESMT", \
          "D6_H7_derived_Sclrtm", "D5_H7_D5CntrlDrmmtm","D2_H7_D2LtM","D3_H7_Cardiac","D2.25_H7_dreived_Somitomere"]

    # select samples 
    D0 = labels[labels["Sample "] == "D0_H7_hESC" ]
    D1 = labels[labels["Sample "] == "D1_H7_derived_APS" ]

    D2 = labels[labels["Sample "] == "D2_H7_derived_DLL1pPXM" ]
    D3 = labels[labels["Sample "] == "D3_H7_derived_ESMT" ]
    D2_25 = labels[labels["Sample "] == "D2.25_H7_dreived_Somitomere" ]


    D5 = labels[labels["Sample "] == "D5_H7_D5CntrlDrmmtm" ]
    D6 = labels[labels["Sample "] == "D6_H7_derived_Sclrtm" ]

    labels2 = pd.concat((D0,D1,D2,D2_25,D3,D5,D6),ignore_index=True)

    #----

    le = preprocessing.LabelEncoder()
    le.fit(labels2["Sample "])
    ttt = le.transform(labels2["Sample "])
    s = pd.DataFrame(ttt)
    ss = pd.concat([labels2["GEO_ID"],s], axis=1, ignore_index=True)
    ss.columns =["Sample_ID","Class_label"]
    samples =list(ss["Sample_ID"])

    #---Select samples 
    selected_data = Trans_data.loc[samples]
    selected_data_gene = Gene_data.loc[samples]

    #print(ss.shape)
    #print(labels2.shape)
    print(selected_data.shape)
    print(selected_data_gene.shape)
    #=============================================================================================
    # needed
    TX_SET= set([int(Trans_names[tx_cnt][4:]) for tx_cnt in range(len(Trans_names))])    # set of all transcripts IDs as integers 
    path_to_ref='/home/maburid/BioinfAlgoProject_spring_2019/Homo_sapiens.GRCh38.rel79.cdna.all.utr_mod.fa'    # change this 
    [TX_to_ENSG,ENSG_isoforms]=readENS_ids(path_to_ref)


    
    
    
    
    
    
    

    
