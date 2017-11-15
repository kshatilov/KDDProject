import os, sys
sys.path.append('../')
import numpy as np
import pandas as pd
import scipy as sp
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as skCluster
from sklearn import datasets as skDatasets
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import FastICA as skFastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn import preprocessing as skPreprocessing
from sklearn import metrics as skMetrics
#from sklearn.metrics import pairwise_distances
#from sklearn.cluster import KMeans
import time
from SubKMeans import SubKMeans


def runEvalMetrics(X, labels_true, labels_pred, f):
	half_run = labels_pred.shape[0]
	nmi_score = np.zeros(half_run)
	fmi_score = np.zeros(half_run)
	sil_score = np.zeros(half_run)

	for i in range(half_run):
		nmi_score[i] = skMetrics.normalized_mutual_info_score(labels_true, labels_pred[i])
		fmi_score[i] = skMetrics.fowlkes_mallows_score(labels_true, labels_pred[i])
		sil_score[i] = skMetrics.silhouette_score(X, labels_pred[i], metric='euclidean')

	f.write("===== Summary =====\n")
	f.write ("NMI score : " + str( np.sum(nmi_score)/float(half_run) ) + "\n" )
	f.write ("FMI score : " + str( np.sum(fmi_score)/float(half_run) ) + "\n" )
	f.write ("Silhouette Coefficient : " + str( np.sum(sil_score)/float(half_run) ) + "\n" )


def removeHalfUpperCosts(costs, labels_pred, numDataPnt):
	full_run = labels_pred.shape[0]
	half_run = int(full_run/2)
	mid_cost = np.partition(costs, half_run-1)[half_run-1]
	labels_half = np.zeros( (half_run, numDataPnt) )
	cnt = 0
	for j in range(full_run):
		if ( costs[j] <= mid_cost ):
			labels_half[cnt] = labels_pred[j]
			cnt += 1
			if ( cnt >= half_run):
				break
	return labels_half


def runPCA(log_name):
	# preprocess
	log_name = log_name + '_PCA.txt'
	f = open(log_name, 'w')
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)
	# params
	numClass = len(np.unique(y))
	numDataPnt = X_scaled.shape[0]
	numDimen = X_scaled.shape[1]
	# run PCA
	start_time = time.time()
	pca = skPCA(n_components=numClass)
	X_pca = pca.fit(X_scaled).transform(X_scaled)
	costs = np.zeros(NUM_RUN)
	labels_pred = np.zeros( (NUM_RUN, numDataPnt) )
	labels_half = np.zeros( (int(NUM_RUN/2), numDataPnt) )
	for i in range(NUM_RUN):
		kmeans_model = skCluster.KMeans(n_clusters=numClass, init='random').fit(X_pca)
		costs[i] = kmeans_model.inertia_
		labels_pred[i] = kmeans_model.labels_
	end_time = time.time()
	labels_half = removeHalfUpperCosts(costs, labels_pred, numDataPnt)
	# run Evaluations
	runEvalMetrics(X_pca, labels_true=y, labels_pred=labels_half, f=f)
	f.write("\n")
	f.write("# of Class : %d, # of Data Points : %d, # of Dimensions : %d \n" %(numClass, numDataPnt, numDimen))
	f.write("Shape of X [%d %d]" %(X_pca.shape[0], X_pca.shape[1]))
	f.write('\n')
	f.write("Clustering took %.2f s\n" %(end_time - start_time) )
	f.close()


def runFastICA(log_name):
	# preprocess
	log_name = log_name + '_ICA.txt'
	f = open(log_name, 'w')
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)
	# params
	numClass = len(np.unique(y))
	numDataPnt = X_scaled.shape[0]
	numDimen = X_scaled.shape[1]
	# run ICA
	start_time = time.time()
	ica = skFastICA(n_components=numClass)
	X_ica = ica.fit(X_scaled).transform(X_scaled)
	costs = np.zeros(NUM_RUN)
	labels_pred = np.zeros( (NUM_RUN, numDataPnt) )
	labels_half = np.zeros( (int(NUM_RUN/2), numDataPnt) )
	for i in range(NUM_RUN):
		kmeans_model = skCluster.KMeans(n_clusters=numClass, init='random').fit(X_ica)
		costs[i] = kmeans_model.inertia_
		labels_pred[i] = kmeans_model.labels_
	end_time = time.time()
	labels_half = removeHalfUpperCosts(costs, labels_pred, numDataPnt)
	# run Evaluations
	runEvalMetrics(X_ica, labels_true=y, labels_pred=labels_half, f=f)
	f.write("\n")
	f.write("# of Class : %d, # of Data Points : %d, # of Dimensions : %d \n" %(numClass, numDataPnt, numDimen))
	f.write("Shape of X [%d %d]" %(X_ica.shape[0], X_ica.shape[1]))
	f.write('\n')
	f.write("Clustering took %.2f s\n" %(end_time - start_time) )
	f.close()

def runLDA(log_name):
	# preprocess
	log_name = log_name + '_LDA.txt'
	f = open(log_name, 'w')
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)
	# params
	numClass = len(np.unique(y))
	numDataPnt = X_scaled.shape[0]
	numDimen = X_scaled.shape[1]
	# run PCA
	start_time = time.time()
	lda = skLDA(n_components=numClass)
	X_lda = lda.fit(X_scaled, y).transform(X_scaled)
	costs = np.zeros(NUM_RUN)
	labels_pred = np.zeros( (NUM_RUN, numDataPnt) )
	labels_half = np.zeros( (int(NUM_RUN/2), numDataPnt) )
	for i in range(NUM_RUN):
		kmeans_model = skCluster.KMeans(n_clusters=numClass, init='random').fit(X_lda)
		costs[i] = kmeans_model.inertia_
		labels_pred[i] = kmeans_model.labels_
	end_time = time.time()
	labels_half = removeHalfUpperCosts(costs, labels_pred, numDataPnt)
	# run Evaluations
	runEvalMetrics(X_lda, labels_true=y, labels_pred=labels_half, f=f)
	f.write("\n")
	f.write("# of Class : %d, # of Data Points : %d, # of Dimensions : %d \n" %(numClass, numDataPnt, numDimen))
	f.write("Shape of X [%d %d]" %(X_lda.shape[0], X_lda.shape[1]))
	f.write('\n')
	f.write("Clustering took %.2f s\n" %(end_time - start_time) )
	f.close()

def runSubKmeans(log_name):
	# preprocess
	log_name = log_name + '_SubKMeans.txt'
	f = open(log_name, 'w')
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)
	# params
	numClass = len(np.unique(y))
	numDataPnt = X_scaled.shape[0]
	numDimen = X_scaled.shape[1]
	# run Kmeans
	start_time = time.time()
	costs = np.zeros(NUM_RUN)
	labels_pred = np.zeros( (NUM_RUN, numDataPnt) )
	labels_half = np.zeros( (int(NUM_RUN/2), numDataPnt) )
	for i in range(NUM_RUN):
		subkmeans = SubKMeans(n_clusters=numClass).fit(X_scaled)
		costs[i] = subkmeans.cost_function_value
		labels_pred[i] = subkmeans.labels_
	end_time = time.time()
	labels_half = removeHalfUpperCosts(costs, labels_pred, numDataPnt)
	# run Evaluations
	runEvalMetrics(X_scaled, labels_true=y, labels_pred=labels_half, f=f)
	f.write("\n")
	f.write("# of Class : %d, # of Data Points : %d, # of Dimensions : %d \n" %(numClass, numDataPnt, numDimen))
	f.write("Shape of X [%d %d]" %(X_scaled.shape[0], X_scaled.shape[1]))
	f.write('\n')
	f.write("Clustering took %.2f s\n" %(end_time - start_time) )
	f.close()

# Load datasets
PATH_DATA = '../data/'
f_pendigits = PATH_DATA + 'pendigits/pendigits.tra'
f_seeds = PATH_DATA + 'seeds/seeds_dataset.txt'
f_symbol_test = PATH_DATA + 'Symbols/Symbols/Symbols_TEST.arff'
f_oliveoil = PATH_DATA + 'OliveOil/OliveOil/OliveOil.arff'

# f_wine = PATH_DATA + 'wine/wine/wine.arff'
f_ecoli = PATH_DATA + 'ecoli/ecoli.txt'
f_soybean = PATH_DATA + 'soybean/soybean.csv'
f_plane = PATH_DATA + 'Plane/Plane.arff'
f_stickfigures = PATH_DATA + 'stickfigures/stickfigures.arff'


f_seeds_log = 'seeds_log'
f_pendigits_log = 'pendigits_log'
f_symbols_log = 'symbols_log'
f_oliveoil_log = 'oliveoil_log'

f_wine_log = 'wine_log'
f_ecoli_log = 'ecoli_log'
f_soybean_log = 'soybean_log'
f_plane_log = 'plane_log'
f_stickfigures_log = 'stickfigures_log'

NUM_RUN = 2


print (" =========== Evaluation for Wines ========== ")
data_wine = skDatasets.load_wine()
X = data_wine.data
y = data_wine.target
df = pd.DataFrame(np.hstack(( X, np.matrix(y).T )))
runPCA(f_wine_log)
runFastICA(f_wine_log)
runLDA(f_wine_log)
runSubKmeans(f_wine_log)

# print (" =========== Evaluation for Ecoli ========== ")
# df = pd.read_csv(f_ecoli, delim_whitespace=True, header=None)
# runPCA(f_ecoli_log)
# runFastICA(f_ecoli_log)
# runLDA(f_ecoli_log)
# runSubKmeans(f_ecoli_log)

# print (" =========== Evaluation for Seeds ========== ")
# df = pd.read_csv(f_seeds, delim_whitespace=True, header=None)
# runPCA(f_seeds_log)
# runFastICA(f_seeds_log)
# runLDA(f_seeds_log)
# runSubKmeans(f_seeds_log)

# print (" =========== Evaluation for Pendigits ========== ")
# df = pd.read_csv(f_pendigits, sep=',', header=None)
# runPCA(f_pendigits_log)
# runFastICA(f_pendigits_log)
# runLDA(f_pendigits_log)
# runSubKmeans(f_pendigits_log)

# print (" =========== Evaluation for Soybean ========== ")
# df = pd.read_csv(f_soybean, sep=',', header=None)
# runPCA(f_soybean_log)
# runFastICA(f_soybean_log)
# runLDA(f_soybean_log)
# runSubKmeans(f_soybean_log)

# print (" =========== Evaluation for Symbols ========== ")
# dataset = arff.loadarff( f_symbol_test )
# df = pd.DataFrame(dataset[0])
# runPCA(f_symbols_log)
# runFastICA(f_symbols_log)
# runLDA(f_symbols_log)
# runSubKmeans(f_symbols_log)

# print (" =========== Evaluation for OliveOil ========== ")
# dataset = arff.loadarff( f_oliveoil )
# df = pd.DataFrame(dataset[0])
# runPCA(f_oliveoil_log)
# runFastICA(f_oliveoil_log)
# runLDA(f_oliveoil_log)
# runSubKmeans(f_oliveoil_log)

# print (" =========== Evaluation for Plane ========== ")
# dataset = arff.loadarff( f_plane )
# df = pd.DataFrame(dataset[0])
# runPCA(f_plane_log)
# runFastICA(f_plane_log)
# runLDA(f_plane_log)
# runSubKmeans(f_plane_log)

# print (" =========== Evaluation for stickfigures ========== ")
# dataset = arff.loadarff( f_stickfigures )
# df = pd.DataFrame(dataset[0])
# runPCA(f_stickfigures_log)
# runFastICA(f_stickfigures_log)
# runLDA(f_stickfigures_log)
# runSubKmeans(f_stickfigures_log)