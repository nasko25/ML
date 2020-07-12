import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use("ggplot")

X = np.array([
		[1, 3],
		[2.1, 2.4],
		[6.7, 9],
		[1.12, 0.8],
		[10, 11.2],
		[12, 15],
	])
# 			  # all the 0th elements from X
# 			  #   ↓
# plt.scatter(X[:,0], X[:,1], s=50)			# s - size
# 			  # 		  ↑
# 			  # all the 1st elements from X
# plt.show()

# The number of clusters is how many types the data can be split into 
classifier = KMeans(n_clusters = 2)

classifier.fit(X)

centroids = classifier.cluster_centers_		# the central point of the cluster
labels = classifier.labels_					# which cluster the data point belongs to

colors = ["g.", "r."]

for i in range(len(X)):		# labels will be 0 or 1, as there are only 2 clusters
							# 			  ↓
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 7)

plt.scatter(centroids[:,0], centroids[:,1], c = "cyan", marker = "x", s = 50, linewidths = 2)
plt.show()

print(X)
print(X[:,0])