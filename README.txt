1. Requirements:
   Program was developed and tested under Python 3.6.2 :: Anaconda, Inc.

2. How to execute:
   To run  subKMeans with wine dataset use following command:
        python main.py
   To use subKMeans with arbitrary dataset (as 2D array named X)
   in Python code use (similar to sklearn.kmeans):
        subkmeans = SubKMeans(n_clusters=3).fit(X)
        #clustered labels are accessible through
        subkmeans.labels_

3. The description of each source file:
   Utilities.py
        File containing supplementary function rvs() for generating random orthogonal matrices
   SubKMeans.py
        File containing source code of algorithm. Most functions has short descriptions.
   main.py
        File containing simple examples of dataset reading and clustering.

4. The operating system, where program was tested:
   Windows 10 Home

5. Additionally
   ---