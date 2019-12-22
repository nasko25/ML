import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class support_vector_machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            
    # train        
    def fit(self, data):
        self.data = data
        # {||w|| : [w, b]} dictionary { key(magnitude of w) : value(list of w and b) }
        opt_dict = {}
        
        # apply the transforms to the vector w
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,1]]
                                          
        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
		# support vectors yi(xi.w+b) = 1
		# choose a constant that is very near 1 to stop optimizing
		
		
        step_sizes = [self.max_feature_value * 0.1,     # big steps
                      self.max_feature_value * 0.01,    # smaller steps
                      # point of expense:
                      self.max_feature_value * 0.001]     # to make more precise, just add smaller step sizes
        
        # extremely expensive
        b_range_multiple = 5    # does not have to be as presise as w
        
        # we don't need to take as small of steps 
		# with b as we do w
        b_multiple = 5
       
        #saves a lot of processing; but less accurate
        latest_optimum = self.max_feature_value * 10
        
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            
            # we can do this because convex (we will know when it is optimized)
            optimized = False
            
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), 
									self.max_feature_value*b_range_multiple,
									step*b_multiple):		# like python range(); np.arange(first value in the range, last value, step size)
					for transformation in transforms:
						w_t = w*transformation
						found_option = True
						# This is the weakest link in the SVM funndamentally;
						# SMO attempts to fix this a bit
						# (run this calculation on all of the data to make sure it fits)
						# Constraint function: yi*(xi.w+b) >= 1
						for i in self.data:
							for xi in self.data[i]:
								yi = i
								if not yi*(np.dot(w_t, xi) + b) >= 1:
									found_option = False
									# TODO should break here; as constraint function fails (so this yi does not work)
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t, b]		# np.linalg.norm(w_t) - get the magnitude of the w_t vector
			
				# the two ws are identical, so choose one to compare
				if w[0] < 0:
					# not the best value, but works
					optimized = True
					print("Optimized a step.")
				else: 
					# w = [5, 5]
					# step = 1
					# w - step = [4, 4]
					w = w - step
					
			norms = sorted([n for n in opt_dict])	# sort from lowest to highest
			
			# the w dictionary looks like this:
			# ||w|| : [w, b]
			# magnitude of w : [w,b]
			# key : value
			opt_choice = opt_dict[norms[0]]			# smallest norm 
			
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimum = opt_choice[0][0] + step*2
		
       
    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        
        return classification


data_dictionary = {-1:np.array([[1, 7], 
                                [2, 8], 
                                [3, 8]]), 
                    1:np.array([[5, 1], 
                                [6, -1], 
                                [7, 3]])}
                                
                               