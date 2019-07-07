import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from math import sqrt

style.use("fivethirtyeight")

plot1 = [1, 3]
plot2 = [2, 5]

def euclidean_distance(plot1, plot2):
	if len(plot1) != len(plot2):
		return -1
	
	sum = 0
	for i in range(0, len(plot1)):
		sum = sum + (plot1[i] - plot2[i])**2
	return sqrt(sum)
	
print(euclidean_distance(plot1, plot2))

r'''
C:\Users\Atanas Pashov\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.7>"Python 3.7 (32-bit).lnk" "C:\Users\Atanas Pashov\OneDrive\Programming\Machine Learning Tutorials\euclidean_distance.py"
2.23606797749979
'''


dataset = {
		'k':[
				[1, 2], 
				[2, 3], 
				[3, 1]
			], 
		'r':[
				[6, 5], 
				[7, 7], 
				[8, 6]
			]
		}
		
new_features = [5, 7]

def visualize_points():
	'''for i in dataset:
		for ii in dataset[i]:
			plt.scatter(ii[0], ii[1], s=100, color = i)
	^ this is the same as:
	'''
	[[plt.scatter(ii[0], ii[1], s=100, color = i) for ii in dataset[i]] for i in dataset]

	plt.scatter(new_features[0], new_features[1])

	plt.show()
	
# visualize_points()

def k_nearest_neighbors(data, predict, k = 3):
	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups!")
	
	# knnalgos
	return vote_result

print(len(dataset))