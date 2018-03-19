import numpy as np
import random
# from sklean import train_test_split
data = np.loadtxt('sin.csv', delimiter = ',')
from scipy.special import expit

## This is only going to work for a binary outpput and the the last layer needs to be 1 neuron
class NN(object):

	def __init__ (self, data, labels, size = [2,2,4,1], alpha = .1, activation_funtion = 'sig', batch = 20, test_percentage = .3):
		# self.data = self.add_ones(data)
		self.data = data
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, labels, test_size=test_percentage, random_state=42)
		self.size = size
		self.alpha = alpha
		self.batch = batch   ##this isn't being used right now.  Will update later
		self.weights, self.bias = self.create_weights()
		self.activation_funtion = activation_funtion ##I'm also going to update this later


	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def sig_prime(self, sig):
		return sig*(1-sig)
		#input should be the result of the activation funtion, not the neuron 


	def create_weights(self):
		a,b = np.shape(self.data)
		w = [np.random.randn(y,x) for x,y in zip(self.size[:-1], self.size[1:])]
		b = [np.random.randn(1,y) for y in self.size]
		return w, b

	def feed_foward(self, data, weights, bias):
		temp = data
		#record activations of each
		act = {}
		count = 0
		for w,b in zip(weights, bias):
			temp = self.sigmoid(np.inner(temp, w)+b)
			act[count] = temp
			count += 1
		return temp, act

	def back_prop(self, data, labels, act, weights):
		pass

x = data[:, 0:2]
y = data[:, 2:3]
a = NN(x, y)
c,b = a.feed_foward(a.data, a.weights, a.bias)

# print a.back_prop(c,labels,b)



