# find the k-smallest
import heapq 
import itertools
import operator
find_len = len(Costs)/ 2
Results_wh = heapq.nsmallest(find_len, range(len(Costs)), np.array(Costs).take)
print(find_len, range(len(Costs)), Results_wh)
Big_Results = pd.concat([Results_gt, Results[Results.columns[Results_wh]]], axis=1)
Big_Results.head()
