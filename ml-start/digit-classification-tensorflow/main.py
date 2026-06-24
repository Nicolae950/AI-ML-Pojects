import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
# baza de date mnsit ce contine cifrele scrise de mana
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# incarcarea dateleor mnist ,
# x_train reprezinta cele 60000 de intrari de 28 x 28 cu coeficientii culorii, fiind datele de antrenare
# y_train reprezinta cifrele reale ale celor 10000 de intrari din x_train, pentru compararea rezultatelor
# x_test reprezinta cele 10000 de intrari de 28 x 28 cu coeficientii culorii, fiind datele de testare
# y_test reprezinta cifrele reale ale celor 10000 de intrari din x_test, pentru compararea rezultatelor

x_train = tf.keras.utils.normalize(x_train, axis=1)
print(len(x_train))
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(len(x_test))

# Crearea retelei neuronale
nr_neuroni_hidden_1 = 128
nr_neuroni_hidden_2 = 128

activare = tf.nn.sigmoid

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# stratul de intrare de 784 de neuroni de intrare

model.add(tf.keras.layers.Dense(units=128, activation=activare))
# primul hidden layer de 16 neuroni cu functia de activare sigmoidala
model.add(tf.keras.layers.Dense(units=128, activation=activare))
# al doilea hidden layer de 16 neuroni cu functia de activare sigmoidala

model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
# output layer de 10 neuroni cu functia de activare softmax
# Functia de activare softmax incearca sa i-a toate output-urile, face scalarea valorilor si ofera un
# un procentaj de probabilitate ca un anumit input sa reprezinte cifra reprezentata

opt_grad_desc = tf.keras.optimizers.SGD(learning_rate=0.01)
opt_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt_grad_desc, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss: " + str(loss) + "%  ||   Acuratete: " + str(accuracy * 100) + "%")

model.save('digits.model')