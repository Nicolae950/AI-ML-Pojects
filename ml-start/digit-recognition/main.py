import matplotlib.pyplot as plt
from data import get_mnist_train, get_mnist_test
import numpy as np
import time

images_train, labels_train = get_mnist_train()  # Incarcarea datelor din MNIST
print(len(images_train), len(labels_train))

images_test, labels_test = get_mnist_test()
print(len(images_test), len(labels_test))

# Se face definirea retelei neuronale
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))  # Initializarea tariilor sinaptice intre stratul input -> hidden
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))  # Initializarea tariilor sinaptice intre stratul hidden -> output

# initialiare neuroni bias
b_i_h = np.zeros((20, 1))
# initializarea legaturilor sinaptice a neur bias conectati intre stratul input -> hidden
b_h_o = np.zeros((10, 1))  # initializarea legaturilor sinaptice a neur bias intre stratul hidden -> output


learn_rate = 0.01
nr_correct = 0
#nr_pas = 1
epochs = 5

# Invatare retea
for epoch in range(epochs):
    for img, lab in zip(images_train, labels_train):
        img.shape += (1,)
        lab.shape += (1,)

        #print(f"Invatare => Epoca : {epoch + 1} Pasul : {nr_pas} / 60000 ", end="\r")
        #time.sleep(0.05)

        # Forward prop. input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        # Forward prop. hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - lab) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(lab))

        # Backprop. output -> hidden (cost function derivative)
        delta_o = o - lab
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backprop. hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

        #nr_pas = nr_pas + 1

    # Show accuracy for this epoch
    print(f"\nTrain Accuracy of epoch {str(epoch + 1)} : {round((nr_correct / images_train.shape[0]) * 100,2)} %")
    nr_correct = 0
    #nr_pas = 1

# Testare Acuratete
for img,lab in zip(images_test,labels_test):
    img.shape += (1,)
    lab.shape += (1,)

    #print(f"Testare => Pasul : {nr_pas} / 10000 ", end="\r")
    #time.sleep(0.05)

    # Forward prop. input -> hidden
    h_pre = b_i_h + w_i_h @ img
    h = 1 / (1 + np.exp(-h_pre))

    # Forward prop. hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    # Cost / Error calculation
    e = 1 / len(o) * np.sum((o - lab) ** 2, axis=0)
    nr_correct += int(np.argmax(o) == np.argmax(lab))

    #nr_pas = nr_pas + 1

print(f"\nTest Accuracy: {round((nr_correct / images_test.shape[0]) * 100, 2)} %")
nr_correct = 0
#nr_pas = 1

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images_train[index]
    plt.imshow(img.reshape(28,28), cmap="Greys")

    img.shape += (1,)

    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))

    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f" Is it : {o.argmax()} ")
    plt.show()