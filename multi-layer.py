import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


data = np.loadtxt('train_MNIST.csv', dtype = str, delimiter = ',')
y = np.asarray(data[1:, 0:1], dtype='float')
X = np.asarray(data[1:,1:], dtype='float')




#from andrew ng's course
a,b = np.shape(X)
input_layer_size = b-1
hiden_layer_size = 4
num_labels = len(np.unique(y))

print num_labels




#dont' need to define a funtin - just a placehold
def sigmoid(z):
	return expit(z)
