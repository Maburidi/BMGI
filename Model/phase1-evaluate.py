"""
This script evaluates the classification potential of gene sets on a dataset.
"""
import argparse
import numpy as np
import pandas as pd
import random
import sys
import utils
import json
import os
from operator import add
import operator
from sklearn.metrics import confusion_matrix
import statistics
from collections import Counter
import sklearn

def evaluate_curated(data, labels, clf, name, genes, cv=5, n_jobs=None, verbose=True, outfile=None):
	# evaluate gene set
	scores,cm = utils.evaluate_gene_set_with_cm(data, labels, clf, genes, cv=cv, n_jobs=n_jobs)
	# compute stats
	n, mu, sigma, Max, Min = len(scores), np.mean(scores), np.std(scores), max(scores), min(scores)
	if verbose:
		print("%-40s %3d %8.3f %8.3f %8.3f %8.3f" % (name, n, mu, sigma, Max, Min))
	print(cm)
	# write results to output file
	if outfile:
		outfile.write("%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n" % (name, n, mu, sigma, Max, Min))
	return cm 


def evaluate_random(data, labels, clf, n_genes, n_iters=100, cv=5, n_jobs=None, verbose=True, outfile=None):
	# evaluate n_iters random sets
	scores = []  
	sum_prp = [] 
	for i in range(n_iters):   #n_iters
		# generate random gene set
		genes = random.sample(list(data.columns), n_genes)

		# evaluate gene set and get predictions for confusion matrix
		scores += utils.evaluate_gene_set(data, labels, clf, genes, cv=cv, n_jobs=n_jobs)
		y_pred= utils.evaluate_gene_set_with_cm_for_random(data, labels, clf, genes, cv=cv, n_jobs=n_jobs)       
		if i==0:
			x = [[] for i in range(len(y_pred))]
		for i in range(len(y_pred)):
			x[i].append(y_pred[i])       

	y_pred_all =[]
	for i in range(len(x)):
		c = Counter(x[i])
		mode_x = c.most_common(1)[0][0]
		y_pred_all.append(mode_x)
        #y_pred_all.append(statistics.mode(x[i]))
    #y_prop_final =[]
	#for i in range(len(sum_prp)):
	#	newList =[]
	#	newList[:] = [x/n_iters for x in sum_prp[i]]
	#	y_prop_final.append(newList)
        
	#y_pred =[]
	#for i in range(len(y_prop_final)):
	#	index = max(enumerate(y_prop_final[i]), key=operator.itemgetter(1))
	#	y_pred.append(index[0])
       
	conf_mat = confusion_matrix(labels, np.asarray(y_pred_all))
	cm = []
	for i in range(len(conf_mat[0])):
		cm.append(list(conf_mat[i]))

	# compute stats
	n, mu, sigma, Max, Min = len(scores), np.mean(scores), np.std(scores), max(scores), min(scores)

	# print results
	if verbose:
		print("%-40s %3d %8.3f %8.3f %8.3f %8.3f" % (str(n_genes), n, mu, sigma, Max, Min))
	print(cm)
	# write results to output file
	if outfile:
		outfile.write("%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n" % (str(n_genes), n, mu, sigma, Max, Min))
	return cm 
 

if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser(description="Evaluate classification potential of gene sets")
	parser.add_argument("--dataset", help="input dataset (samples x genes)", required=True)
	parser.add_argument("--labels", help="list of sample labels", required=True)
	parser.add_argument("--model-config", help="model configuration file (JSON)", required=True)
	parser.add_argument("--model", help="classifier model to use", default="mlp-tf")
	parser.add_argument("--outfile", help="output file to save results")
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--full", help="Evaluate the set of all genes in the dataset", action="store_true")
	parser.add_argument("--random", help="Evaluate random gene sets", action="store_true")
	parser.add_argument("--random-range", help="range of random gene sizes to evaluate", nargs=3, type=int, metavar=("START", "STOP", "STEP"))
	parser.add_argument("--random-iters", help="number of iterations to perform for random classification", type=int, default=100)
	parser.add_argument("--n-jobs", help="number of parallel jobs to use", type=int, default=None)
	parser.add_argument("--cv", help="number of folds for k-fold cross validation", type=int, default=5)
	parser.add_argument("--cm", help="print confusion matrix- path for output json file is required")

    
    
	args = parser.parse_args()

	# load input data
	print("loading input dataset...")

	df = utils.load_dataframe(args.dataset)
	df_samples = df.index
	df_genes = df.columns

	labels = utils.load_labels(args.labels)

	print("loaded input dataset (%s genes, %s samples)" % (df.shape[1], df.shape[0]))

	# initialize classifier
	print("initializing classifier...")

	clf = utils.load_classifier(args.model_config, args.model)

	print("initialized %s classifier" % args.model)

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		curated_sets = utils.load_gene_sets(args.gene_sets)

		print("loaded %d gene sets" % (len(curated_sets)))

		# remove genes which do not exist in the dataset
		genes = list(set(sum([genes for (name, genes) in curated_sets], [])))
		missing_genes = [g for g in genes if g not in df_genes]

		curated_sets = [(name, [g for g in genes if g in df_genes]) for (name, genes) in curated_sets]

		print("%d / %d (%0.1f%%) genes from gene sets were not found in the input dataset" % (
			len(missing_genes),
			len(genes),
			len(missing_genes) / len(genes) * 100))
	else:
		curated_sets = []

	# include the set of all genes if specified
	if args.full:
		curated_sets.append(("FULL", df_genes))

	# initialize list of random set sizes
	if args.random:
		# determine random set sizes from range
		if args.random_range != None:
			print("initializing random set sizes from range...")
			random_sets = range(args.random_range[0], args.random_range[1] + 1, args.random_range[2])

		# determine random set sizes from gene sets
		elif args.gene_sets != None:
			print("initializing random set sizes from curated sets...")
			random_sets = sorted(set([len(genes) for (name, genes) in curated_sets]))

		# print error and exit
		else:
			print("error: --gene-sets or --random-range must be provided to determine random set sizes")
			sys.exit(1)
	else:
		random_sets = []

	print("evaluating gene sets...")

	# initialize output file
	if args.outfile:
		outfile = open(args.outfile, "w")
		outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("name", "n", "mu", "sigma","max", "min"))

	# evaluate curated gene sets
	dic_cm={}    
	for (name, genes) in curated_sets:
		cm = evaluate_curated(df, labels, clf, name, genes, cv=args.cv, n_jobs=args.n_jobs, outfile=outfile)
		dic_cm.update({name:str(cm)}) 


	# evaluate random gene sets
	for n_genes in random_sets:
		cm_r=evaluate_random(df, labels, clf, n_genes, n_iters=args.random_iters, cv=args.cv, n_jobs=args.n_jobs, outfile=outfile)
		dic_cm.update({str(n_genes):str(cm_r)}) 
        
        
	if args.cm:
		with open(str(args.cm), 'w') as fp:
			json.dump(dic_cm, fp)
        
        
        
        
