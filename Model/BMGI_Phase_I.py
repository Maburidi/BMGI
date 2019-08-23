"""
This script implements phase I in BMGI algorithm: more detials can be found in BMGI paper: 

Random Forest is used here to score every gene for classification power using its \
isoforms expression values as features.

Required Inputs:
1- Annotated gene sets. Each gene is represented by an Ensemble ID (e.g. ENSG00000159251). txt file
2- TEM: sigle-cell processed transcriptomic expression data (rows are the samples (i.g. cells), columns are the transcripts). csv file
3- GEM: sigle-cell processed gene expression data (rows are the samples (i.g. cells), columns are the genes). csv file
4- Sample labels (samples (i.g. cells) belongs to classes, each samples should have a label). csv file

Not required inputs:
5- P-value threshold (defult = 0.001).
6- Percentage threshold (defult = 0.6). 
7- Number of trees for Random Forest (defult =100). 
8- Number of folds for cross validation (defult =10). 

Outputs:
json file contains the Annotated gene sets, each set has  genes that have significant classification power to random gene. 
The jason file contains all of the statistics of each gene.    



# Example:

python Model/BMGI_Phase_I.py \
    --gene_sets    /home/maburid/AMIA_project/msigdb.v6.2.symbols.txt \
    --TEM_dataset  /home/maburid/AMIA_project/Transc_stem_cell.csv \
    --GEM_dataset  /home/maburid/AMIA_project/Gene_stem_cell.csv \
    --labels       /home/maburid/AMIA_project/labels.csv \
    --num_trees    50 \
    --cv           5 \
    --outfile /home/maburid/AMIA_project/

"""


import gc
gc.collect()
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.mixture
import scipy
import argparse
import copy

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
    parser.add_argument("--percent_threshold", help="minimum percentage of filtered genes in a set to the origional genes, otherwise, if less than this threshould, dont count the set as a significant", type=float, default=0.6)
    parser.add_argument("--outfile", help="output dir to save results",required=True )

    
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
    
    rand_score = []
    
    # get accuracy for random gene
    for u in range(0,100):
        gene = random.choice(list(ENSG_isoforms.keys()))    
        isoforms=ENSG_isoforms[str(gene)]
        ## Data for the model   
        X = selected_data[isoforms]
        Y = list(ss["Class_label"])  
        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=args.num_trees)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)       
        clf = copy.deepcopy(clf)                                                                               
        clf.fit(X_train, y_train)                                                                         
        scores = clf.score(X_test, y_test)                                                               
        rand_score.append(scores)                      
                                                                                                              
    rand_mean = np.mean(rand_score)                                                                 
    rand_std = np.std(rand_score)     
    rand_max = np.max(rand_score)    
    rand_min = np.min(rand_score)
    
    ## Run the Model: 
    
    thresh_p_value = args.p_threshold               
    genes = list(ENSG_isoforms.keys())    # 37940 gene  and 172941 transcripts 
                                                                            
    all_infor =[]
    set_name = gene_sets[u][0]
    gene_accuracy ={}  
    two_max_isoforms ={}                            
    C_V = args.cv                  # num of folds for cv 

    for i in range(0,len(genes)): 
        #if i == 50:              
        #    break 
        #if str(genes[i]) in list(ENSG_isoforms.keys()):
        #================================ apply RF on gene transcripts 
        isoforms=ENSG_isoforms[str(genes[i])]
    
        ## Data for the model 
        X = selected_data[isoforms]
        Y = list(ss["Class_label"])
            #print("Data_shape (num_sample, num_isoforms):",X.shape)
            #print("Number_of_samples: ",len(Y))    
        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=args.num_trees)
        clf.fit(X, Y)
        scores = sklearn.model_selection.cross_val_score(clf, X, y=Y, cv=C_V)
            #y_pred_prob = sklearn.model_selection.cross_val_predict(clf, X, y=Y, cv=5 )
        isoform_score = list(clf.feature_importances_ )
        
        scores_for_plot = []
        sel_mean = np.mean(scores)
        sel_std = np.std(scores)
        sel_max = np.max(scores)
        sel_min = np.min(scores)
    
        scores_for_plot.append(sel_mean)
        scores_for_plot.append(sel_std)
        scores_for_plot.append(sel_max)
        scores_for_plot.append(sel_min)
        
        #============================ get p_value 
        t,p_value = scipy.stats.ttest_ind_from_stats(rand_mean, rand_std, 100, sel_mean, sel_std, C_V, equal_var=False)
        scores_for_plot.append(p_value)
        gene_accuracy[str(genes[i])] = scores_for_plot
                    
        #============================
            # this is to get the maximum accurate isoforms. 
        max_isofomes =[] 
        index, value = max(enumerate(isoform_score), key=operator.itemgetter(1))
        max_isofomes.append(index)
        isoform_score.remove(value)
        if len(isoform_score) > 0:
            index, value = max(enumerate(isoform_score), key=operator.itemgetter(1))
            max_isofomes.append(index)
        max_isofomes = [ isoforms[i] for i in max_isofomes ]
        two_max_isoforms[genes[i]] = max_isofomes
        #=============================
        print(i)
        i = i+1
    
    
    informative_genes ={} 
    for ggene in gene_accuracy:
        if gene_accuracy[ggene][0] > rand_mean and gene_accuracy[ggene][4] < thresh_p_value :
            informative_genes[ggene] =gene_accuracy[ggene]
    print(informative_genes)
        
    #all_infor.append(informative_genes) 
    #all_infor.append(len(list(informative_genes.keys()))/len(genes))                  # save the percent of the informative genes
    #sig_sets[set_name] = all_infor
    two_infor_isoforms ={}

    for two_iso in list(two_max_isoforms.keys()):
        if two_iso in list(two_infor_isoforms.keys()):
            continue
        two_infor_isoforms[two_iso] = two_max_isoforms[two_iso]

    #=============================================================
    # Group
    sig_sets = {}

    for u in range(0,len(gene_sets)):                           
        #if u == 16:                            
        #    break                                   
    
        all_infor =[]             
        set_name = gene_sets[u][0]          
        genes = symb_to_id(gene_sets[u][2:])   
        #print("===================================================================")
        #print("Set Name: ",set_name)
        print("Set Number: ", u)
        #print("Number of Genes:", len(genes))
        get_sig_genes= {}
        combine_prec =[] 
        for i in range(len(genes)):
            if genes[i] in list(informative_genes.keys()):
                get_sig_genes[str(genes[i])] = informative_genes[str(genes[i])]
        combine_prec.append(get_sig_genes)
        if len(genes) >0:
            combine_prec.append(len(list(get_sig_genes.keys()))/len(genes))                
        else: 
            combine_prec.append(int(0))    
        sig_sets[str(set_name)] = combine_prec                              
    
                                                                                        
    json2 = json.dumps(sig_sets)                                         
    ff = open( args.outfile + "BMGI_Phase_I_results.json","w")             
    ff.write(json2)                                                                                                       
    ff.close()  
    
   
    

    
    
    

    


