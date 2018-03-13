import numpy as np
import random
# from sklean import train_test_split
data = np.loadtxt('sin.csv', delimiter = ',')
from scipy.special import expit


class NN(object):

	def __init__ (self, data, labels, size = [2,2,4,1], alpha = .1, batch = 20, test_percentage = .3):
		# self.data = self.add_ones(data)
		self.data = data
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, labels, test_size=test_percentage, random_state=42)
		self.size = size
		self.alpha = alpha
		self.batch = batch
		self.weights, self.bias = self.create_weights()


	def signoid(self, x):
		return expit(x)

	def sig_prime(self, sig):
		return sig*(1-sig)
		#just to be clear, instead of computing sigmoid 3 times, i can just store the value, and then use it again for the derivative

		
	def add_ones(self, x):
	 	a, b = np.shape(x)
		c = np.ones((a , 1))   
		return np.hstack((c, x))

	def create_weights(self):
		a,b = np.shape(self.data)
		new_size = [b]+self.size
		w = [np.random.randn(y,x) for x,y in zip(new_size[:-1], new_size[1:])]
		b = [np.random.randn(1,y) for y in self.size]
		return w, b

	def feed_foward(self, data, weights, bias):
		temp = data
		for w,b in zip(weights, bias):
			temp = self.signoid(np.inner(temp, w)+b)
			print temp
		return temp


data = np.array(([2,3],[3,4]))
labels = 1

a = NN(data, labels)

a.feed_foward(a.data, a.weights, a.bias)

