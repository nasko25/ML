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
                      [1,-1]]
                                          
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
		# higher number for b_range_multiple will be more precise 
        b_range_multiple = 2 #5    # does not have to be as presise as w
        
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
                                    step*b_multiple):        # like python range(); np.arange(first value in the range, last value, step size)
                                    
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
                                    #print(xi,":",yi*(np.dot(w_t, xi) + b))
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]        # np.linalg.norm(w_t) - get the magnitude of the w_t vector
            
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
                    
            norms = sorted([n for n in opt_dict])    # sort from lowest to highest
            
            # the w dictionary looks like this:
            # ||w|| : [w, b]
            # magnitude of w : [w,b]
            # key : value
            opt_choice = opt_dict[norms[0]]            # smallest norm 
            
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
            
        '''for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi,":",yi*(np.dot(self.w, xi) + self.b))'''
        
       
    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization: 
            self.ax.scatter(features[0], features[1], s = 200, marker = "*", c = self.colors[classification])
        
        return classification

    # not needed for the SVM; it is only for humans to see :)
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s = 100, color = self.colors[i]) for x in data_dictionary[i]] for i in data_dictionary]
        
        # hyperplane = x.w+b
        # v = x.w+b
        # positive support vector = 1
        # negative support vector = -1
        # decision boundary = 0
        def hyperplane(x, w, b, v):            # v - the values we are seeking
            return (-w[0]*x-b+v) / w[1]
            
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)        # psv1 - just a scalar value (y)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], color = "k")        # black
        
        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)        # psv1 - just a scalar value (y)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], "k")                # black
        
        # (w.x+b) = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)        # psv1 - just a scalar value (y)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], "y--")        # yellow dashes

        plt.show()


data_dictionary = {-1:np.array([[1, 7], 
                                [2, 8], 
                                [3, 8]]), 
                    1:np.array([[5, 1], 
                                [6, -1], 
                                [7, 3]])}
                                
svm = support_vector_machine()
svm.fit(data = data_dictionary)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]
# really fast predictions once we have w and b
for p in predict_us:
    svm.predict(p)

svm.visualize()