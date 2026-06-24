import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# Pentru Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint

# get the data and pre-processed it

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def plot_input_img(i):
    plt.imshow(X_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()


# Normalizing the image to [0,1] range
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Reshape / Expand the dimentions of images to (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Define the model of neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(48, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))

#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

optim = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optim, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

# EarlyStopping
#es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=4, verbose=1)

# Model Check Point
#mc = ModelCheckpoint("./bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
#cb = [es, mc]

# Model Training
#his = model.fit(X_train, y_train, epochs=5, validation_split=0.1, callbacks=cb)

# Load Model
#model_S = keras.models.load_model("C://Users//colit//PycharmProjects"
#                                  "//HandwrittenDigitClassificationUsingCNN//bestmodel.h5")
#loss, accuracy = model_S.evaluate(X_test, y_test)

model.fit(X_train, y_train, epochs=10, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nThe model accuracy is {str(accuracy * 100)} % \n\nThe loss of model is : {str(loss)}")

#while True:
#    index = int(input("Enter a number (0 - 9999): "))
#    predictions = model.predict(X_test)
#    predictions = tf.nn.softmax(predictions)
#    pred = predictions[index]
#    label = np.argmax(pred)
#    plt.imshow(X_train[index], cmap='binary')
#    plt.title(label)
