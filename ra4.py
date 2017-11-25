import tensorflow as tf
import keras
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

i_train = []
o_train = []

xc = 4
yc = 3

sample = 100

for i in range(sample):
	x = uniform(10)-2
	y = uniform(10)-2
	if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < 2.0:
		c = 1.0
	else:
		c = 0.0

	i_train.append([x, y])
	o_train.append(c)

print(i_train)
print(o_train)

#

xp = []
yp = []
for i in range(sample):
	if o_train[i] < 0.5:
		xp.append(i_train[i][0])
		yp.append(i_train[i][1])

plt.scatter(xp, yp, c="blue", marker="*", s=100)

xn = []
yn = []
for i in range(sample):
	if o_train[i] >= 0.5:
		xn.append(i_train[i][0])
		yn.append(i_train[i][1])

plt.scatter(xn, yn, c="red", marker="*", s=200)
#plt.show()

#import sys
#sys.exit()


from keras.layers import Dense, Activation
model=keras.models.Sequential()

hid_units =3 # +4
act =  'sigmoid' # 'relu' #

model.add(Dense(units=hid_units, input_dim=2))
model.add(Activation(act))

#for i in range(2-1):
#	model.add(Dense(units=hid_units, input_dim=hid_units))
#	model.add(Activation(act))

model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(i_train, o_train, epochs=50000) #50000)

#score = model.evaluate(x_test, y_test, batch_size=1)
#print("accuracy=", score[1])

points = 40
a=[]
for ix in range(points):
	for iy in range(points):
		a.append([ix*10.0/points-1, iy*10.0/points-1])

p = np.array(a)

r = model.predict(p)
print(r)

thresh = 0.5

xp = []
yp = []
for i in range(points ** 2):
	if r[i][1] < thresh:
		xp.append(p[i][0])
		yp.append(p[i][1])

plt.scatter(xp, yp, c="blue", marker=".")

xn = []
yn = []
for i in range(points ** 2):
	if r[i][1] >= thresh:
		xn.append(p[i][0])
		yn.append(p[i][1])

plt.scatter(xn, yn, c="red", marker=".")
plt.show()


