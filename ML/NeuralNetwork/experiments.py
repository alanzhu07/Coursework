import neural_network as NN
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train, X_val, y_val) = NN.load_data_large()
num_epoch = 100
init_rand = True

# Training for different number of hidden units
num_hidden = [5,20,50,100,200]
learning_rate = 0.01
avg_loss_train = [0 for i in range(len(num_hidden))]
avg_loss_valid = [0 for i in range(len(num_hidden))]
for i in range(len(num_hidden)):
    (loss_per_epoch_train, loss_per_epoch_val, 
    err_train, err_val, y_hat_train, y_hat_val) = NN.train_and_valid(
        X_train, y_train, X_val, y_val, num_epoch, num_hidden[i], init_rand, learning_rate)
    print("Done training with {} hidden layers, with avg training loss {}, avg validation loss {}".format(num_hidden[i], loss_per_epoch_train[-1], loss_per_epoch_val[-1]))
    avg_loss_train[i] = loss_per_epoch_train[-1]
    avg_loss_valid[i] = loss_per_epoch_val[-1]
plt.plot(num_hidden, avg_loss_train)
plt.plot(num_hidden, avg_loss_valid)
plt.ylabel("Cross-entropy")
plt.xlabel("Hidden units")
plt.legend(["Training", "Validation"])
plt.savefig('HiddenUnits.png')
plt.clf()

# Training for different learning rates
learning_rate = [0.1, 0.01, 0.001]
num_hidden = 50
epochs = np.arange(1,num_epoch+1)
for i in range(len(learning_rate)):
    (loss_per_epoch_train, loss_per_epoch_val, 
    err_train, err_val, y_hat_train, y_hat_val) = NN.train_and_valid(
        X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate[i])
    print("Done training with learning rate {}".format(learning_rate[i]))
    plt.plot(epochs, loss_per_epoch_train)
    plt.plot(epochs, loss_per_epoch_val)
    plt.ylabel("Cross-entropy")
    plt.xlabel("Epochs")
    plt.title("learning rate = {}".format(learning_rate[i]))
    plt.legend(["Training", "Validation"])
    plt.savefig('LearningRate{}.png'.format(i))
    plt.clf()