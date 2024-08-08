from sklearn.neural_network import MLPClassifier

X = [[1.,0.],
     [2.,0.],
     [3.,0.],
     [0.,1.],
     [0.,2.]]

T = [0, 0, 1, 0, 1]

net = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
                    hidden_layer_sizes=(5), random_state = 1)

net.fit(X, T)

Y = net.predict(X)

print(Y)

print(net.coefs_) # weights?

print(net.intercepts_) # biases?