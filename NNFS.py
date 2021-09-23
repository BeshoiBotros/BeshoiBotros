import matplotlib.pyplot as plt 
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
	return x * (1 - x)

a = [
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,0,0,0,0,1,
]

b = [
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1
]

a = np.array(a).reshape(15,2)

b = np.array(b).reshape(15,2)

x = [a, b]

x = np.array(x)
y = np.array([
	[1,0],
	[0,1]
])

lr = 0.1

w1 = np.random.randn(4,15)
w2 = np.random.randn(2,4)

for i in range(1000):
	z1 = np.dot(w1, x[1])
	a1 = sigmoid(z1)

	z2 = np.dot(w2, a1)
	a2 = sigmoid(z2)

	d2 = (a2 - y)
	d1 = np.dot(w2.T, d2) * d_sigmoid(z1)

	w1_adj = np.dot(d1, x[1].T)
	w2_adj = np.dot(d2, a1.T)

	w1 -= lr * w1_adj
	w2 -= lr * w2_adj

print(w1,'\n\n')
print(w2)
