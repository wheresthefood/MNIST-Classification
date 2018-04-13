import numpy as np
import random


#only for binary distinction
class NN(object):

	def __init__ (self, data, labels, hidden_layer_size = 3, alpha = .1, iterations = 100000, batch = 20):
		self.data = data
		self.alpha = alpha
		self.iter = iterations
		#self.batch = batch
		#I'm going to keep this in here to implement batch gradient decent later
		self.size = hidden_layer_size
		self.wh, self.wo, self.bh, self.bo = self.create_weights()


	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def sig_prime(self, sig):
		return sig*(1-sig)
		#just to be clear, instead of computing sigmoid 3 times, i can just store the value, and then use it again for the derivative


	def create_weights(self):
		a,b = np.shape(self.data)
		wh = np.random.randn(b, self.size)
		wo = np.random.randn(self.size,1)
		bh = np.random.randn(1, self.size)
		bo = np.random.randn(1, 1)
		return wh, wo, bh, bo

 	def feed_foward(self, data, wh, wo, bh, bo):

 		output1 = self.sigmoid(np.dot(data, wh)+bh)
 		output2 = self.sigmoid(np.dot(output1, wo)+bo)

 		return output1, output2

 	def back_prop(self, data, labels, output1, output2, wh, wo, bh, bo, alpha = 1):
 		error = labels - output2
 		delta_output = error*self.sig_prime(output2)
 		error_hidden = delta_output.dot(wo.T)
 		delta_hidden = error_hidden*self.sig_prime(output1)
 		wo +=  alpha*output1.T.dot(delta_output)
 		wh += alpha*data.T.dot(delta_hidden)
 		bo += alpha*np.sum(delta_output, axis = 0)
 		bh += alpha*np.sum(delta_hidden, axis = 0)
 		total_error = np.sum(abs(error))
 		return wo, wh, bh, bo, total_error


 	def train(self, data, labels, wh, wo, bh, bo, alpha = 100, iter = 120000):
 		error = []
 		for i in range(0, iter):
 			output1, output2 = self.feed_foward(data,wh, wo, bh, bo)
 			wo, wh, bh, bo, total_error = self.back_prop(data, labels, output1, output2, wh, wo, bh, bo, alpha = alpha)
 			if i%1000 == 0:
 				if i%10000:
 					print total_error, i//1000, 'Thousand Iterations'
 				error.append(total_error)
 		return wh, wo, bh, bo, total_error

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