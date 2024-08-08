import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


(xTrain, tTrain), (xTest, tTest) = mnist.load_data()

xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1).astype('float32') / 255
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1).astype('float32') / 255
tTrain = to_categorical(tTrain, 10)
tTest = to_categorical(tTest, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, tTrain, batch_size=128, epochs=10, verbose=1, validation_data=(xTest, tTest))

score = model.evaluate(xTest, tTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


yPred = model.predict(xTest)
yPred_classes = np.argmax(yPred, axis=1)
yTrue = np.argmax(tTest, axis=1)

conf_matrix = confusion_matrix(yTrue, yPred_classes)


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
