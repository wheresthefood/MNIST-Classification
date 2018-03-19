import numpy as np
import random
# from sklean import train_test_split
data = np.loadtxt('sin.csv', delimiter = ',')
from scipy.special import expit


#only for binary distinction

class NN(object):

	def __init__ (self, data, labels, hidden_layer_size = 3, alpha = .1, batch = 20, test_percentage = .3):
		self.data = data
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, labels, test_size=test_percentage, random_state=42)
		self.alpha = alpha
		self.batch = batch
		self.size = hidden_layer_size
		self.wh, self.wo, self.bh, self.bo = self.create_weights()


	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def sig_prime(self, sig):
		return sig*(1-sig)
		#just to be clear, instead of computing sigmoid 3 times, i can just store the value, and then use it again for the derivative


	def create_weights(self):
		a,b = np.shape(self.data)
		#wh equals hidden layer
		#wo equal output layer
		wh = np.random.randn(b, self.size)
		wo = np.random.randn(self.size,1)
		bh = np.random.randn(1, self.size)
		bo = np.random.randn(1, 1)
		return wh, wo, bh, bo

 	def feed_foward(self, data, wh, wo, bh, bo):
 		output1 = self.sigmoid(np.dot(data, wh)+bh)
 		output2 = self.sigmoid(np.dot(output1, wo)+bo)

 		return output1, output2

 	#this will work for one training example
 	def back_prop(self, data, labels, output1, output2, wh, wo, bh, bo, alpha = 1):
 		error = labels - output2
 		delta_output = error*self.sig_prime(output2)
 		# print np.shape(delta_output)
 		# print np.shape(wo)
 		error_hidden = delta_output.dot(wo.T)
 		delta_hidden = error_hidden*self.sig_prime(output1)
 		wo +=  alpha*output1.T.dot(delta_output)
 		wh += alpha*data.T.dot(delta_hidden)
 		bo += alpha*np.sum(delta_output, axis = 0)
 		bh += alpha*np.sum(delta_hidden, axis = 0)

 		total_error = np.sum(abs(error))
 		return wo, wh, total_error

 	# def GD(self, data, labels, wh, wo, bh, bo):
 	# 	#code from https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/
		# a, b = feed_foward(self, data, wh, wo, bh, bo)

		# params = [wh, wo, bh, bo]

		# grads = T.grad(cost=cost, wrt=params)
		# updates = []

		#   for p, g in zip(params, grads):
		#     updates.append([p, p - g * lr])

		#   return updates

# updates = sgd(cost, params)



 	def train(self, data, labels, wh, wo, bh, bo, alpha = 100, iter = 120000):

 		for i in range(0, iter):
 			output1, output2 = self.feed_foward(data,wh, wo, bh, bo)
 			wo, wh, total_error = self.back_prop(data, labels, output1, output2, wh, wo, bh, bo, alpha = alpha)
 			if i%1000 == 0: print total_error
 		return wh, wo, bh, bo

 	def test(self, data, labels, wh, wo, bh, bo):
 		out1, out2 = self.feed_foward(data,wh, wo, bh, bo)
 		def pr(x):
 			output = []
 			for i in x:
 				if i >=.5:
 					output.append(1)
 				else:
 					output.append(0)
 			return np.asarray([output]).T
 		correct = pr(out2)==labels
		return correct*1, pr(out2), out2


x = data[:, 0:2]
y = data[:, 2:3]


# x = np.array([[0,0,1],
#             [0,1,1],
#             [1,0,1],
#             [1,1,1]])
                
# y = np.array([[0],
# 			[1],
# 			[1],
# 			[0]])
a = NN(x, y)

c,d = a.feed_foward(x, a.wh, a.wo, a.bh, a.bo)

w1,w2, bh, bo = a.train(x, y,a.wh, a.wo, a.bh, a.bo, alpha = .0005, iter = 200000)


c,d,e   = a.test(x, y, w1, w2, bh, bo)
# print len(c)
print sum(c)
print sum(d)

