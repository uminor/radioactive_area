#
# "radioactive_area.py"
#
#  an example of keras (with tensorflow by Google)
#   by U.minor
#    free to use with no warranty
#
# usage:
# python radioactive_area.py 10000
#
# last number (10000) means learning epochs, default=1000 if omitted

import tensorflow as tf
import keras
from keras.optimizers import SGD
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import sys
import time

argvs = sys.argv

i_train, o_train = [], []

xc, yc = 4.0, 3.0

sample = 100

# generate sample data
for i in range(sample):
	x = uniform(10) - 2
	y = uniform(10) - 2
	if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < 2.0:
		c = 1.0	# dangerous point
	else:
		c = 0.0	# safe point

	i_train.append([x, y])
	o_train.append(c)

print(i_train)
print(o_train)

xps, yps = [], []
for i in range(sample):
	if o_train[i] < 0.5:
		xps.append(i_train[i][0])
		yps.append(i_train[i][1])

xns, yns = [], []
for i in range(sample):
	if o_train[i] >= 0.5:
		xns.append(i_train[i][0])
		yns.append(i_train[i][1])

## if plot input image
plt.scatter(xps, yps, c="blue", marker="*", s=100)
plt.scatter(xns, yns, c="red", marker="*", s=200)
plt.show()

#sys.exit()


from keras.layers import Dense, Activation
model = keras.models.Sequential()

# neural network model parameters
hidden_units = 3
layer_depth = 1
act =  'sigmoid' # 'relu' #

# first hidden layer
model.add(Dense(units = hidden_units, input_dim = 2, use_bias=True))
model.add(Activation(act))

# additional hidden layers (if necessary)
for i in range(layer_depth - 1):
	model.add(Dense(units = hidden_units, input_dim = hidden_units, use_bias=True))
	model.add(Activation(act))

# output layer
model.add(Dense(units = 2, use_bias=True))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# training
if len(argvs) > 1 and argvs[1] != '':
	ep = int(argvs[1]) # from command line
else:
	ep = 1000

start_fit = time.time()
model.fit(i_train, o_train, epochs = ep, verbose = 1)
elapsed = time.time() - start_fit
print("elapsed = {:.1f} sec".format(elapsed))

#score = model.evaluate(x_test, y_test, batch_size = 1)
#print("accuracy=", score[1])

# predict
ticks = 40
a = []
for ix in range(ticks):
	for iy in range(ticks):
		a.append([ix * 10.0 / ticks - 1.0, iy * 10.0 / ticks - 1.0])

p = np.array(a)

r = model.predict(p)
print(r)

thresh = 0.5

# safe area (blue points)
xp, yp = [], []
for i in range(ticks ** 2):
	if r[i][1] < thresh:
		xp.append(p[i][0])
		yp.append(p[i][1])

plt.scatter(xp, yp, c="cyan", marker=".")

# dangerous area (red points)
xn, yn = [], []
for i in range(ticks ** 2):
	if r[i][1] >= thresh:
		xn.append(p[i][0])
		yn.append(p[i][1])

plt.scatter(xn, yn, c="magenta", marker=".")

plt.scatter(xps, yps, c="blue", marker="*", s=100)
plt.scatter(xns, yns, c="red", marker="*", s=200)

plt.show()


