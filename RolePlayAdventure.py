##################################################################
# New project #
##################################################################
# K-means cluster #
##################################################################

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

#Plotting and visualizing our data befor feeding it into the Machine Learning Algorithm // for Anzahl Cluster
x1 = [1, 5, 1.5, 8, 1, 9]
y1 = [2, 8, 1.8, 8, 0.6, 11]
plt.scatter(x1, y1)
plt.show()

# Converting our data to a NumPy array
X1 = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Wie initialize K-means algorithm with the required parameter and wie use .fit() to fit the data
kmeans = KMeans(n_clusters=2) # Angabe, wie viele cluster wir haben wollen
kmeans.fit(X1)

# Getting the values of centroids and labels based on the fitment
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

#Plotting and visualizing output

colors = ['g.', 'r.', 'c.', 'y.']
for i in range(len(x1)):
    print('coordinate:', X1[i], 'label:', labels[i])
    plt.plot(X1[i][0], X1[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5)

plt.show()