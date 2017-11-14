1. How to execute:
   To run evaluation code with wine dataset use following command:
        cd eval
        python YoungRunEvaluations.py
   Then output log files will be generated in current folder(=eval)
        1. wine_log_SubKMeans.txt
        2. wine_log_PCA.txt
        3. wine_log_ICA.txt
        4. wine_log_LDA.txt

2. The description of each source file:
   YoungRunEvaluations.py
        Main evaluation code for running experiments for various datasets on several metrics (NMI, FMI, Silhouette, Running Time)
   YoungScale.py
        Code for preprocessing the datasets
   calMetrics_ORCLUS.py
        Code for calculating metrics mentioned above from results of ORCLUS clustering algorithm.
   *.ipynb
        Example codes for basic functionalities.

3. The operating system, where program was tested:
   Windows 10 Home

4. Additionally
...