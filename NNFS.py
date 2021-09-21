import numpy as np 

def sigmoid(x):
	return 1 / (np.exp(-x))

_1 = [
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
]
_2 = [
	0,1,1,1,0,
	0,1,0,0,1,
	0,0,0,1,0,
	0,0,1,0,0,
	0,0,1,0,0,
	1,1,1,1,1
]

x = [np.array(_1).reshape(1,30), np.array(_2).reshape(1,30)]

y = [
	[1,0],
	[0,1]
]
y = np.array(y)

lr = 0.1

w1 = np.zeros((30,4))
w2 = np.zeros((4,2))

z1 = np.dot(x, w1)
a1 = sigmoid(z1)

z2 = np.dot(a1, w2)
a2 = sigmoid(z2)



for i in range(100):
	#error
	d2 = (a2 - y)
	d1 = np.dot(w2, d2.T).T * (a1 * (1 - a1))

	w1_adj = np.dot(d1, x)
	w2_adj = np.dot(d2, a1)

	w1 -= lr * w1_adj
	w2 -= lr * w2_adj