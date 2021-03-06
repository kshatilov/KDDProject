from sklearn import metrics
from sklearn import datasets
from SubKMeans import SubKMeans
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing as skPreprocessing
from scipy.io import arff
import time


# Example of using with dataset packed in .csv
def read_data():
    df = pd.read_csv("data/soybean/soybean.csv", sep=';', encoding='', header=None)
    X = np.array(df.iloc[:, 1:])
    labels = np.ndarray.flatten(np.array(df.iloc[:, 0:1]))
    return X, labels


# Example of using with dataset packed in .arff
def read_symbols():
    dataset = arff.loadarff("symbols.arff")
    df = pd.DataFrame(dataset[0])
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    return X, y


# Method of visualising results (scatterplot)
def visualize_results(X, labels):
    data = pd.DataFrame(data=X)
    color_wheel = {1: "#0392cf",
                   2: "#7bc043",
                   3: "#ee4035"}
    colors = pd.Series(labels).apply(lambda x: color_wheel.get(x + 1))
    scatter_matrix(data, alpha=0.2, figsize=(15, 15), color=colors, diagonal='kde')
    plt.show()


# Complete example of usage.
# Non-commented code proceeds clustering over wine dataset
def main():
    # X, labels = read_symbols()
    dataset = datasets.load_wine()
    X = dataset.data
    labels = dataset.target
    X = skPreprocessing.scale(X)
    # Perform clustering
    start_time = time.time()
    subkmeans = SubKMeans(n_clusters=3).fit(X)
    print("SubKMeans: Clustering took: " + str(time.time() - start_time) + " s")
    print("SubKMeans: NMI score: " + str(metrics.normalized_mutual_info_score(subkmeans.labels_, labels)))
    # Get scatter plot for output of the algorithm
    # visualize_results(np.dot(X, subkmeans.V), subkmeans.labels_)


if __name__ == "__main__":
    main()
