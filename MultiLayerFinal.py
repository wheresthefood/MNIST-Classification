import numpy as np
import random


#only for binary distinction
class NN(object):

	def __init__ (self, data, labels, hidden_layer_size = 3):
		self.data = data
		self.labels = labels
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

 	def feed_foward(self, data):
		output1 = self.sigmoid(np.dot(data, self.wh)+self.bh)
		output2 = self.sigmoid(np.dot(output1, self.wo)+self.bo)


 		return output1, output2

 	def back_prop(self, output1, output2, alpha):
 		error = self.labels - output2
 		delta_output = error*self.sig_prime(output2)
 		error_hidden = delta_output.dot(self.wo.T)
 		delta_hidden = error_hidden*self.sig_prime(output1)
 		self.wo +=  alpha*output1.T.dot(delta_output)
 		self.wh += alpha*self.data.T.dot(delta_hidden)
 		self.bo += alpha*np.sum(delta_output, axis = 0)
 		self.bh += alpha*np.sum(delta_hidden, axis = 0)
 		total_error = np.sum(abs(error))
 		return total_error


 	def train(self, data = None, alpha = 100, iterations = 120000):
 		data = self.data
 		error = []
 		for i in range(0, iterations):
 			output1, output2 = self.feed_foward(data)
 			total_error = self.back_prop(output1, output2, alpha)
 			if i%1000 == 0:
 				if i%10000:
 					print total_error, i//1000, 'Thousand Iterations'
 				error.append(total_error)
 		return total_error

 	def predict(self, x):
 		out1, out2 = self.feed_foward(x)
 		results = []
 		for i in out2:
			if i >=.5:
				results.append(1)
			else:
				results.append(0)
		return np.asarray([results]).T

	def test(self, x, labels):
		results =  self.predict(x)
		correct = (results==self.labels)*1
		return correct, results