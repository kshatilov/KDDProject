from Utilities import rvs
import numpy as np
import math
import random

"""

Python implementation of SubKMeans clustering algorithm
All credits go to authors of the paper "Towards an Optimal Subspace for K-Means"
Dominik Mautz, Wei Ye, Claudia Plant, Christian Böhm
KDD’17, August 13–17, 2017, Halifax, NS, Canada

"""


class SubKMeans:
    # Calculates dataset (or cluster) mean
    @staticmethod
    def calculate_dataset_mean(X):
        if len(X) <= 0:
            return 0
        mean = np.zeros(len(X[0]))
        for x in X:
            mean = np.add(mean, x)
        for m in np.nditer(mean, op_flags=['readwrite']):
            m[...] = m / len(X)
        return mean

    # Calculates scatter matrix of dataset (or cluster)
    @staticmethod
    def calculate_scatter_matrix(x):
        d = len(x)
        if d == 0:
            return [0]
        c = np.identity(d) - np.multiply(1. / d, np.ones((d, d)))
        s_d = np.dot(x.T, c)
        s_d = np.dot(s_d, x)
        return s_d

    # Returns projection matrix onto first m attributes
    @staticmethod
    def get_projection_matrix(m, d):
        p = np.identity(m)
        zeros = np.zeros((m, d - m))
        p = np.append(p, zeros, 1)
        return p

    # Returns projection matrix for last (d-m) attributes
    @staticmethod
    def get_p_n(m, d):
        p = np.identity(d - m)
        zeros = np.zeros((d - m, m))
        p = np.append(zeros, p, 1)
        return p

    # Returns n random datapoints from given dataset
    @staticmethod
    def get_random_datapoints(X, n):
        return random.choices(X, k=n)

    # Evaluates simplified cost function for given datapoint and cluster mean
    @staticmethod
    def cost_function_i(_x, _cluster_mean):
        return math.pow(np.linalg.norm(_x - _cluster_mean), 2)

    # Evaluates cost function for given datapoint and cluster mean
    # DEPRECATED
    def cost_function(self, x, cluster_mean):
        _x = np.dot(self.pT, self.V.T)
        _x = np.dot(_x, x)
        _cluster_mean = np.dot(self.pT, self.V.T)
        _cluster_mean = np.dot(_cluster_mean, cluster_mean)
        return math.pow(np.linalg.norm(_x - _cluster_mean), 2)

    # Performs eigendecomposition of scatter matrices
    def eig(self):
        sm = np.zeros((self.d, self.d))
        for si in self.scatter_mx:
            sm = np.add(sm, si)
        sm = np.subtract(sm, self.S_D)
        return np.linalg.eig(sm)

    # Changes the number of dimensions of clustering subspace
    # to the amount of negative eigenvalues
    def updateM(self, E):
        count = 0
        for e in E:
            if e < 0:
                count = count + 1
        return count

    # Calculates cost function, which is supposed to be minimised.
    # If value of cost_function is not changed through iterations, function returns true, meaning convergence
    def convergence(self):
        total_error = 0.
        for error in self.cluster_errors:
            total_error = total_error + error

        # second part of cost function
        _p_n = SubKMeans.get_p_n(self.m, self.d)
        _phi_D = np.dot(_p_n, self.V.T)
        _phi_D = np.dot(_phi_D, self.phi_D)
        for x in self.X:
            _x = np.dot(_p_n, self.V.T)
            _x = np.dot(_x, x)
            total_error = \
                total_error + \
                self.cost_function_i(_x, _phi_D)

        prev = self.cost_function_value
        self.cost_function_value = total_error
        if math.fabs(prev - total_error) < 0.0001:
            return True

    # Initialises class, with given number of clusters
    def __init__(self, n_clusters=2):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError('SubKMeans: Number of clusters should be positive integer')

        self.cluster_weights = []
        self.n_clusters = n_clusters
        self.V = rvs()
        self.m = 0
        self.d = 0
        self.phi_D = []
        self.S_D = []
        self.cluster_means = []
        self.clusters = []
        self.scatter_mx = []
        self.labels_ = []
        self.cluster_errors = []
        self.cost_function_value = 0.
        self.X = []
        self.pT = []

    # Performs (subspace) clustering over given dataset X
    def fit(self, X):
        if len(X) <= 0:
            raise ValueError('SubKMeans: Dataset is corrupted')

        # INITIALIZATION
        self.X = X
        self.d = len(X[0])
        self.m = int(math.floor(math.sqrt(self.d)))
        self.V = rvs(self.d)
        self.phi_D = self.calculate_dataset_mean(X)
        self.S_D = self.calculate_scatter_matrix(X)
        self.cluster_means = self.get_random_datapoints(X, self.n_clusters)
        self.pT = SubKMeans.get_projection_matrix(self.m, self.d)

        # REPEAT
        iterations = 0
        while True:
            # Renew info about clusters
            self.clusters = []
            self.scatter_mx = []
            self.cluster_errors = []
            self.cluster_weights = []

            for i in range(0, self.n_clusters):
                self.clusters.append([])
                self.scatter_mx.append([])
                self.cluster_errors.append(0)
                _cluster_mean = np.dot(self.pT, self.V.T)
                _cluster_mean = np.dot(_cluster_mean, self.cluster_means[i])
                self.cluster_weights.append(_cluster_mean)

            # ASSIGNMENT STEP
            self.labels_ = []
            for x in X:

                _x = np.dot(self.pT, self.V.T)
                _x = np.dot(_x, x)

                min_cost = self.cost_function_i(_x, self.cluster_weights[0])
                closest_cluster = 0
                for i in range(1, self.n_clusters):
                    cost = self.cost_function_i(_x, self.cluster_weights[i])
                    if cost < min_cost:
                        min_cost = cost
                        closest_cluster = i

                self.clusters[closest_cluster].append(x)
                self.labels_.append(closest_cluster)
                self.cluster_errors[closest_cluster] = self.cluster_errors[closest_cluster] + min_cost

            # UPDATE STEP
            for i in range(0, self.n_clusters):
                self.cluster_means[i] = self.calculate_dataset_mean(self.clusters[i])
                self.scatter_mx[i] = self.calculate_scatter_matrix(np.array(self.clusters[i]))

            # EIGENDECOMPOSITION
            E, self.V = self.eig()
            self.m = self.updateM(E)

            # CONVERGENCE
            if self.convergence():
                break

            # Infinite loop termination
            iterations = iterations + 1
            if iterations > 9999:
                print("SubKMeans: Too many iterations, aborting clustering...")
                break

        self.labels_ = np.array(self.labels_)
        return self
