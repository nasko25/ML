import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from math import sqrt
import pandas as pd
import random

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

def visualize_points(result):
	'''for i in dataset:
		for ii in dataset[i]:
			plt.scatter(ii[0], ii[1], s=100, color = i)
	^ this is the same as:
	'''
	[[plt.scatter(ii[0], ii[1], s=100, color = i) for ii in dataset[i]] for i in dataset]

	plt.scatter(new_features[0], new_features[1], color = result)

	plt.show()
	
# visualize_points()

def k_nearest_neighbors(data, predict, k = 3):
	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups!")
	
	distances = []
	for group in data:
		for features in data[group]:
			# eucledean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			eucledean_distance = np.linalg.norm(np.array(features)-np.array(predict)) # same as ^
			distances.append([eucledean_distance, group])
			
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	
	return vote_result, confidence

print(len(dataset))
result = k_nearest_neighbors(dataset, new_features, k = 3)
print(result)
# visualize_points(result)


# load data from a file
df = pd.read_csv("C:\\Users\\atana\OneDrive\Programming\Machine Learning Tutorials\datasets\dataset.data")
# df = pd.read_csv("C:\\Users\Atanas Pashov\OneDrive\Programming\Machine Learning Tutorials\datasets\dataset.data")
df.replace("?", -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
print(df.head())
full_data = df.astype(float).values.tolist()
print(full_data[:10])

print("#"*50 + " shuffled the data " + "#"*50)
# shuffle the data
random.shuffle(full_data)
print(full_data[:10])

# slice the data
test_size = 0.2 # 20%
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] # the fist 80% of full_data
test_data = full_data[-int(test_size*len(full_data)):] # the last 20% of full_data
# print(len(train_data))
# print(len(test_data))

# populate the train_set and test_set dictionaries
for i in train_data:
	train_set[i[-1]].append(i[:-1]) # i[-1] is the last column from the train_set dictionary (the class value, which is either a 2, or a 4)

for i in test_data:
	test_set[i[-1]].append(i[:-1])
	
	
correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote,confidence = k_nearest_neighbors(train_set, data, k=5)
		# print(data)
		if group == vote:
			correct += 1
		else: 
			print(confidence)
		total += 1
		
print("Accuracy: ", correct/total)