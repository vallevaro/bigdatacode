import scipy.io
import os
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import requests
from io import BytesIO

path = './yeastData310.mat'
data = scipy.io.loadmat(path)

X = data['X']

# corrDist = pdist(X, 'correlation')
# clusterTree = AgglomerativeClustering(n_clusters=16, linkage='single', metric='cityblock')
# clusterTree.fit(X)
# clusters = clusterTree.labels_

# fig, axes = plt.subplots(4, 4)
# fig.suptitle('Hierarchical Clustering of Profiles ', y = 1 ,fontsize = 20)
# times = data['times'].reshape(7, )
# for c in range(0, 16):
#     occurences = np.argwhere(clusters == (c))
#     row = c//4
#     col = c%4
#     for occ in occurences:
#         axes[row][col].plot(times, X[occ, :].reshape(7,))
    
# plt.tight_layout(rect=[0, 0, 1, 0.90])
# plt.show()