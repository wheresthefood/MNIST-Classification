from MultiLayerFinal import NN
import numpy as np
import random
from scipy.special import expit


class Multi_Class(NN):
	def __init__ (self, data, labels, hidden_layer_size = 3, alpha = .1, batch = 20):
		NN.__init__(self, data, labels, hidden_layer_size = 3, alpha = .1, batch = 20)





data = np.loadtxt('sin.csv', delimiter = ',')
x = data[:, 0:2]
y = data[:, 2:3]


a = Multi_Class(x,y)

# c,d = a.feed_foward(x, a.wh, a.wo, a.bh, a.bo)

# w1,w2, bh, bo, error = a.train(x, y,a.wh, a.wo, a.bh, a.bo, alpha = .0005, iter = 200)


# c,d,e   = a.test(x, y, w1, w2, bh, bo)
# # print len(c)
# print sum(c)
# print sum(d)


# data = np.loadtxt('sin.csv', delimiter = ',')