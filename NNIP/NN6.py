from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


(xTrain, tTrain), (xTest, tTest) = mnist.load_data()

xTrain = xTrain.reshape(xTrain.shape[0], -1) / 255.0
xTest = xTest.reshape(xTest.shape[0], -1) / 255.0

mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=0.1)

mlp.fit(xTrain, tTrain)

tPred = mlp.predict(xTest)

accuracy = accuracy_score(tTest, tPred)
conf_matrix = confusion_matrix(tTest, tPred)

print(f"Accuracy: {accuracy:.4f}")

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
