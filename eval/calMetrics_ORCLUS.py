# setup
import numpy as np
import pandas as pd
import scipy as sp
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
from scipy.io import arff
import time
# from SubKMeans import SubKMeans
# %matplotlib inline


def runEvalMetrics(X, labels_true, labels_pred, f):
	# print ("runEvalMetrics")
	half_run = 1
	nmi_score = np.zeros(half_run)
	fmi_score = np.zeros(half_run)
	sil_score = np.zeros(half_run)
	f.write("Labels_True:,")
	for j in range( len(labels_true)-1 ):
		f.write( str(labels_true[j])+',')
	f.write( str(labels_true[j])+'\n')

	for i in range( half_run ):
		f.write( str(i+1) + "th_" +"PredictedLabels:,")
		for j in range( len(labels_pred)-1 ):
			f.write( str(int(labels_pred[j]) )+',' )
		f.write( str( int(labels_pred[j]) )  +'\n' )
	f.write('\n')
	for i in range(half_run):
		nmi_score[i] = skMetrics.normalized_mutual_info_score(labels_true, labels_pred)
		fmi_score[i] = skMetrics.fowlkes_mallows_score(labels_true, labels_pred)
		sil_score[i] = skMetrics.silhouette_score(X, labels_pred, metric='euclidean')
		f.write(str(i+1)+','+str(nmi_score[i])+','+str(fmi_score[i])+','+str(sil_score[i])+'\n')
		# print ("i nmi fmi sil", i, nmi_score[i], fmi_score[i], sil_score[i])

	f.write('\n\n')
	f.write("===== Summary =====\n")
	f.write ("NMI score : " + str( np.sum(nmi_score)/float(half_run) ) + "\n" )
	f.write ("FMI score : " + str( np.sum(fmi_score)/float(half_run) ) + "\n" )
	f.write ("Silhouette Coefficient : " + str( np.sum(sil_score)/float(half_run) ) + "\n" )
	# print ("NMI score : ", str( np.sum(nmi_score)/float(half_run) ) )
	# print ("FMI score : ", str( np.sum(fmi_score)/float(half_run) ) )
	# print ("Silhouette Coefficient : ", str( np.sum(fmi_score)/float(half_run) ) )


def computeMetrics(log_name, name):
	log_name = log_name + '_ORCLUS.txt'
	f = open(log_name, 'w')
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)
	# params
	numClass = len(np.unique(y))
	numDataPnt = X_scaled.shape[0]
	numDimen = X_scaled.shape[1]

	label_pred_dict = {}
	labels_pred = np.zeros( numDataPnt )
	for i in range(numClass):
		f_read = open( name+'/cluster_'+str(i)+".txt", 'r' )
		for line in f_read.readlines():
			line  = line.strip().split()
			# print (line)
			if (line[0][0:3] == 'ID='):
				label_pred_dict[int(line[0][3:])-1] = i
				# print ( str( line[0][3:]) + " / " + str(i) )
		f_read.close()

	for i in range(numDataPnt):
		labels_pred[i] = label_pred_dict[i]
		# print ( str(i) + "   " + str(label_pred_dict[i]))

	# run Evaluations
	runEvalMetrics(X_scaled, labels_true=y, labels_pred=labels_pred, f=f)
	f.write("\n")
	f.write("# of Class : %d, # of Data Points : %d, # of Dimensions : %d \n" %(numClass, numDataPnt, numDimen))
	f.write("Shape of X_scaled [%d %d]" %(X_scaled.shape[0], X_scaled.shape[1]))
	f.write('\n')
	f.write("Clustering took     s\n" )
	f.close()


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
	labels_half = np.zeros( (NUM_RUN/2, numDataPnt) )
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

# Load datasets
PATH_DATA = '../../../../1-1/COMP5331_KDDB/project/data/scaled/'
f_wine_w = PATH_DATA + 'wine.csv'
f_ecoli_w = PATH_DATA + 'ecoli.csv'
f_seeds_w = PATH_DATA + 'seeds.csv'
f_pendigits_w = PATH_DATA + 'pendigits_train.csv'
f_soybean_w = PATH_DATA + 'soybean.csv'
# f_symbols_w = PATH_DATA + 'symbols_test.csv'
# f_oliveoil_w = PATH_DATA + 'oliveoil.csv'
# f_plane_w = PATH_DATA + 'plane.csv'
# f_stickfigures_w = PATH_DATA + 'stickfigures.csv'
wine = 'wine'
ecoli = 'ecoli'
seeds = 'seeds'
pendigits = 'pendigits'
soybean = 'soybean'

f_wine_log = 'wine_log'
f_ecoli_log = 'ecoli_log'
f_seeds_log = 'seeds_log'
f_pendigits_log = 'pendigits_log'
f_soybean_log = 'soybean_log'

# f_oliveoil_log = 'oliveoil_log'
# f_symbols_log = 'symbols_log'
# f_plane_log = 'plane_log'
# f_stickfigures_log = 'stickfigures_log'


print (" =========== Evaluation for Wines ========== ")
data_wine = skDatasets.load_wine()
X = data_wine.data
y = data_wine.target
df = pd.DataFrame(np.hstack(( X, np.matrix(y).T )))
computeMetrics(f_wine_log, wine)

print (" =========== Evaluation for Ecoli ========== ")
df = pd.read_csv(f_ecoli_w, sep=',', header=None)
computeMetrics(f_ecoli_log, ecoli)

print (" =========== Evaluation for Seeds ========== ")
df = pd.read_csv(f_seeds_w, sep=',', header=None)
computeMetrics(f_seeds_log, seeds)

print (" =========== Evaluation for Pendigits ========== ")
df = pd.read_csv(f_pendigits_w, sep=',', header=None)
computeMetrics(f_pendigits_log, pendigits)

print (" =========== Evaluation for Soybean ========== ")
df = pd.read_csv(f_soybean_w, sep=',', header=None)
computeMetrics(f_soybean_log, soybean)