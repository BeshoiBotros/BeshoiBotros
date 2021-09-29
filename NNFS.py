import numpy as np
import matplotlib.pyplot as plt

a = [
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,0,0,0,0,1
]

a = np.array(a).reshape(30,1)

a1 = [
	0,1,1,1,1,0,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,0,0,0,0,1
]
a1 = np.array(a1).reshape(30,1)

b = [
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,1
]

b = np.array(b).reshape(30,1)

b1 = [
	1,1,1,1,1,0,
	1,0,0,0,0,1,
	1,1,1,1,1,1,
	1,0,0,0,0,1,
	1,1,1,1,1,0
]

b1 = np.array(b1).reshape(30,1)

x = [a, b, a1, b1]

y1 = np.array([
	[1],
	[0]
])
y2 = np.array([
	[0],
	[1]
])

y3 = np.array([
	[1],
	[0]
])

y4 = np.array([
	[0],
	[1]
])

y = [y1, y2, y3, y4]

w1 = np.zeros((30, 5))
w2 = np.zeros((5, 2))

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def d_sigmoid(x):
	return (sigmoid(x) / (1 + sigmoid(x)))

def MSE(y_hat, y):
	s = (np.square(y_hat - y))
	s = np.sum(s) / len(y)
	return s

def forward(x, w1, w2):
	z1 = np.dot(w1.T, x) 
	a1 = sigmoid(z1)

	z2 = np.dot(w2.T, a1)
	a2 = sigmoid(z2)

	return a1, a2



def backward(x, w1, w2, y, lr ):
	a1, a2 = forward(x, w1, w2)

	# errors
	d2 = (a2 - y)
	d1 = np.multiply(np.dot(w2, d2), (a1 * (1 - a1)))

	w1_adj = np.dot(x, d1.T)
	w2_adj = np.dot(a1, d2.T)

	w1 -= lr * w1_adj
	w2 -= lr * w2_adj

epochs = 1000
loss = []
acc = []

for i in range(epochs):
	l = []
	for j in range(len(x)):
		a1, a2 = forward(x[j], w1, w2)
		backward(x[j], w1, w2, y[j], lr = 0.1)
		l.append((MSE(a2, y[j])))
	loss.append(sum(l) / len(x))
	print("epochs:", i + 1, "======== acc:", (1 - (sum(l) / len(x)))*100)
	acc.append((1 - (sum(l) / len(x)))*100)


def predict(x, w1, w2):
	a1, a2 = forward(x, w1, w2)
	return a2

print(w1, '\n\n\n\n', w2)


print(predict(x[0], w1, w2))

plt.plot(loss)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()

plt.imshow(b_.reshape(6, 5), cmap = 'gray')
plt.show()
