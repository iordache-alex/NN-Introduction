import numpy as np
import matplotlib.pyplot as plt

def activation(x):
    return x


def perceptron(x, w):
    return activation(np.dot(x, w[1:]) + w[0])


def error(y, y_hat):
    return y - y_hat


def gradient_descent(x, e, w, lr):
    u = lr
    w_new = np.zeros(w.shape)
    w_new[0] = w[0] + u * np.sum(e)
    w_new[1:] = w[1:] + u * np.dot(e, x)
    return w_new

errors = []
weights = []

# Training function
def train(x, y, w, epochs, lr):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        y_hat = perceptron(x, w)
        e = error(y, y_hat)
        w = gradient_descent(x, e, w, lr)
        print(f"Error: {np.sum(np.square(e))}")
        errors.append(np.sum(np.square(e)))
        print(f"Weights: {w}")
        weights.append(w.copy())  # Save a copy of weights at each epoch

    return w


inp = np.array([[1], [2], [3]])
outp = np.array([3, 5, 7])
w = np.array([-2, -2])

w = train(inp, outp, w, 15, 0.1)
print("Final weights:", w)

weights = np.array(weights)


w0 = weights[:, 0]
w1 = weights[:, 1]

W0, W1 = np.meshgrid(w0, w1)
E = np.array(errors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(w0, w1, E, c='b', marker='o')

ax.set_xlabel('Weight 0')
ax.set_ylabel('Weight 1')
ax.set_zlabel('Error')

plt.show()
