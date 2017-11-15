import os, sys
sys.path.append('../')
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
from SubKMeans import SubKMeans

def preprocess(name):
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)

	f = open(name, 'w')
	row = X_scaled.shape[0]
	col = X_scaled.shape[1]
	for i in range(row):
		for j in range(col):
			f.write( str(X_scaled[i,j]) )
			f.write(',')
		f.write( str(y[i]) )
		f.write('\n')
	f.close()

def preprocess_no_groundtruth(name):
	X = df.iloc[:,0:-1]
	y = df.iloc[:, -1]
	X_scaled = skPreprocessing.scale(X)

	f = open(name, 'w')
	row = X_scaled.shape[0]
	col = X_scaled.shape[1]
	for i in range(row):
		for j in range(col-1):
			f.write( str(X_scaled[i,j]) )
			f.write(',')
		f.write( str(X_scaled[i,col-1]) )
		f.write('\n')
	f.close()

# Load datasets
PATH_DATA = '../data/'
NEW_PATH = '../data/scaled/'

f_pendigits = PATH_DATA + 'pendigits/pendigits.tra'
f_seeds = PATH_DATA + 'seeds/seeds_dataset.txt'
f_symbol_test = PATH_DATA + 'Symbols/Symbols/Symbols_TEST.arff'
f_oliveoil = PATH_DATA + 'OliveOil/OliveOil/OliveOil.arff'
# f_wine = PATH_DATA + 'wine/wine/wine.arff'
f_ecoli = PATH_DATA + 'ecoli/ecoli.txt'
f_soybean = PATH_DATA + 'soybean/soybean.csv'
f_plane = PATH_DATA + 'Plane/Plane.arff'
f_stickfigures = PATH_DATA + 'stickfigures/stickfigures.arff'


f_seeds_w = NEW_PATH + 'seeds.csv'
f_pendigits_w = NEW_PATH + 'pendigits_train.csv'
f_symbols_w = NEW_PATH + 'symbols_test.csv'
f_oliveoil_w = NEW_PATH + 'oliveoil.csv'
f_wine_w = NEW_PATH + 'wine.csv'
f_ecoli_w = NEW_PATH + 'ecoli.csv'
f_soybean_w = NEW_PATH + 'soybean.csv'
f_plane_w = NEW_PATH + 'plane.csv'
f_stickfigures_w = NEW_PATH + 'stickfigures.csv'

f_seeds_nogt_w = NEW_PATH + 'seeds_nogt.csv'
f_pendigits_nogt_w = NEW_PATH + 'pendigits_nogt_train.csv'
f_symbols_nogt_w = NEW_PATH + 'symbols_nogt_test.csv'
f_oliveoil_nogt_w = NEW_PATH + 'oliveoil_nogt.csv'
f_wine_nogt_w = NEW_PATH + 'wine_nogt.csv'
f_ecoli_nogt_w = NEW_PATH + 'ecoli_nogt.csv'
f_soybean_nogt_w = NEW_PATH + 'soybean_nogt.csv'
f_plane_nogt_w = NEW_PATH + 'plane_nogt.csv'
f_stickfigures_nogt_w = NEW_PATH + 'stickfigures_nogt.csv'

# Run Evaluations for each dataset
print (" =========== Scaling Seeds ========== ")
df = pd.read_csv(f_seeds, delim_whitespace=True, header=None)
preprocess(f_seeds_w)

print (" =========== Scaling Pendigits ========== ")
df = pd.read_csv(f_pendigits, sep=',', header=None)
preprocess(f_pendigits_w)

print (" =========== Scaling Symbol ========== ")
dataset = arff.loadarff( f_symbol_test )
df = pd.DataFrame(dataset[0])
preprocess(f_symbols_w)

print (" =========== Scaling OliveOil ========== ")
dataset = arff.loadarff( f_oliveoil )
df = pd.DataFrame(dataset[0])
preprocess(f_oliveoil_w)

print (" =========== Scaling Wine ========== ")
data_wine = skDatasets.load_wine()
X = data_wine.data
y = data_wine.target
df = pd.DataFrame(np.hstack(( X, np.matrix(y).T )))
preprocess(f_wine_w)

print (" =========== Scaling ecoli ========== ")
df = pd.read_csv(f_ecoli, delim_whitespace=True, header=None)
preprocess(f_ecoli_w)

df = pd.read_csv(f_soybean, sep=',', header=None)
preprocess(f_soybean_w)

dataset = arff.loadarff( f_plane )
df = pd.DataFrame(dataset[0])
preprocess(f_plane_w)

print (" =========== Scaling Stickfigures ========== ")
dataset = arff.loadarff( f_oliveoil )
df = pd.DataFrame(dataset[0])
preprocess(f_stickfigures_w)

# ///////////////////////////////////////////////////////////////////

# Run Evaluations for each dataset
print (" =========== Seeds ========== ")
df = pd.read_csv(f_seeds, delim_whitespace=True, header=None)
preprocess_no_groundtruth(f_seeds_nogt_w)

print (" =========== Pendigits ========== ")
df = pd.read_csv(f_pendigits, sep=',', header=None)
preprocess_no_groundtruth(f_pendigits_nogt_w)

print (" =========== Symbol ========== ")
dataset = arff.loadarff( f_symbol_test )
df = pd.DataFrame(dataset[0])
preprocess_no_groundtruth(f_symbols_nogt_w)

print (" =========== OliveOil ========== ")
dataset = arff.loadarff( f_oliveoil )
df = pd.DataFrame(dataset[0])
preprocess_no_groundtruth(f_oliveoil_nogt_w)

print (" =========== Wine ========== ")
data_wine = skDatasets.load_wine()
X = data_wine.data
y = data_wine.target
df = pd.DataFrame(np.hstack(( X, np.matrix(y).T )))
preprocess_no_groundtruth(f_wine_nogt_w)

print (" =========== ecoli ========== ")
df = pd.read_csv(f_ecoli, delim_whitespace=True, header=None)
preprocess_no_groundtruth(f_ecoli_nogt_w)

df = pd.read_csv(f_soybean, sep=',', header=None)
preprocess_no_groundtruth(f_soybean_nogt_w)

dataset = arff.loadarff( f_plane )
df = pd.DataFrame(dataset[0])
preprocess_no_groundtruth(f_plane_nogt_w)

print (" =========== stickfigures ========== ")
dataset = arff.loadarff( f_stickfigures )
df = pd.DataFrame(dataset[0])
preprocess_no_groundtruth(f_stickfigures_nogt_w)
