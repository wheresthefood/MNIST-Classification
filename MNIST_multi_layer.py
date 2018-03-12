import numpy as np
import random
# from sklean import train_test_split
data = np.loadtxt('sin.csv', delimiter = ',')


class NN(object):

	def __init__ (self, data, labels, size = [2,2,4,1], alpha = .1, batch = 20, test_percentage = .3):
		self.data = self.add_ones(data)
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, labels, test_size=test_percentage, random_state=42)
		self.size = size
		self.alpha = alpha
		self.batch = batch
		self.weights = self.create_weights()


		
	def add_ones(self, x):
	 	a, b = np.shape(x)
		c = np.ones((a , 1))   
		return np.hstack((c, x))

	def create_weights(self):
		a,b = np.shape(self.data)
		new_size = [b]+self.size
		w = [np.random.randn(y,x) for x,y in zip(new_size[:-1], new_size[1:])]
		return w

	def feed_foward(self, data, weights):
		temp = data
		for i in weights:
			temp = np.inner(temp, i)
			print np.shape(temp)
		return temp




# data = np.array(([3,2,3],[2,3,3],[3,4,3],[1,2,3]))
labels = 1

a = NN(data, labels)
for i in a.weights:
	print np.shape(i)

a.feed_foward(a.data, a.weights)

# print a.feed_foward(a.data, a.weights)
