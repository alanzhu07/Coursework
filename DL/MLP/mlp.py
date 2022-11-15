"""
Fall 2022, 10-417/617
Assignment-1

IMPORTANT:
    DO NOT change any function signatures

September 2022
"""


import numpy as np

def random_weight_init(input, output):
    b = np.sqrt(6)/np.sqrt(input+output)
    return np.random.uniform(-b, b, (output, input))

def zeros_bias_init(outd):
    return np.zeros((outd, 1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(14)] for lab in labels])


class Transform:
    """
    This is the base class. You do not need to change anything.

    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass

class ReLU(Transform):
    """
    ReLU non-linearity, combined with dropout
    IMPORTANT the Autograder assumes these function signatures
    """
    def __init__(self, dropout_probability=0):
        Transform.__init__(self)
        self.dropout = dropout_probability

    def forward(self, x, train=True):
        # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function
        """
        x shape (indim, batch_size)
        """
        if train:
            dropout_sampled = np.random.uniform(0, 1, x.shape)
            self.active = dropout_sampled > self.dropout
            self.input = x
            return np.where(self.active, np.maximum(x, 0.), 0.)
        else:
            return (1 - self.dropout) * np.maximum(x, 0.)

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        return np.where(self.active & (self.input > 0), grad_wrt_out, 0.)

class LinearMap(Transform):
    """
    Implement this class
    For consistency, please use random_xxx_init() functions given on top for initialization
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        Transform.__init__(self)
        self.indim, self.outdim = indim, outdim
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)
        self.b = random_weight_init(1, outdim)
        self.gradW = np.zeros(self.W.shape)
        self.gradb = np.zeros(self.b.shape)
        self.momentumW = np.zeros(self.W.shape)
        self.momentumb = np.zeros(self.b.shape)

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.input = x
        self.output = self.b + np.matmul(self.W, x)
        return self.output

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        Your backward call should accumulate gradients.
        """
        grad = np.matmul(self.W.T, grad_wrt_out)
        self.gradW += np.matmul(grad_wrt_out, self.input.T)
        self.gradb += grad_wrt_out.sum(axis=1)[:,None]
        return grad

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        self.momentumW = self.alpha * self.momentumW + self.gradW
        self.momentumb = self.alpha * self.momentumb + self.gradb
        self.W -= self.lr * self.momentumW
        self.b -= self.lr * self.momentumb

    def zerograd(self):
        self.gradW = 0.
        self.gradb = 0.

    def getW(self):
        """
        return W shape (outdim, indim)
        """
        return self.W

    def getb(self):
        """
        return b shape (outdim, 1)
        """
        return self.b

    def loadparams(self, w, b):
        self.W = w
        self.b = b

class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (num_classes,batch_size)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        self.softmax_out = np.exp(logits)
        self.softmax_out /= np.sum(self.softmax_out, axis=0)
        self.cross_entropy_loss = -(labels*np.log(self.softmax_out)).sum(axis=0).mean()
        self.labels = labels
        return self.cross_entropy_loss

    def backward(self):
        """
        return shape (num_classes,batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return -(self.labels - self.softmax_out) / self.labels.shape[1]

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        pred = self.softmax_out.argmax(axis=0)
        true = self.labels.argmax(axis=0)
        return (pred == true).mean()

class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, outdim, hiddenlayer=100, alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.input = LinearMap(indim, hiddenlayer, alpha=alpha, lr=lr)
        self.relu = ReLU(dropout_probability=dropout_probability)
        self.output = LinearMap(hiddenlayer, outdim, alpha=alpha, lr=lr)

    def forward(self, x, train=True):
        """
        x shape (indim, batch_size)
        """
        x = self.input.forward(x)
        x = self.relu.forward(x, train=train)
        x = self.output.forward(x)
        return x

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad_wrt_out = self.output.backward(grad_wrt_out)
        grad_wrt_out = self.relu.backward(grad_wrt_out)
        grad_wrt_out = self.input.backward(grad_wrt_out)
        return grad_wrt_out

    def step(self):
        self.input.step()
        self.output.step()

    def zerograd(self):
        self.input.zerograd()
        self.output.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        self.input.loadparams(Ws[0], bs[0])
        self.output.loadparams(Ws[1], bs[1])

    def getWs(self):
        """
        Return the weights for each layer
        Return a list containing weights for first layer then second and so on...
        """
        return [self.input.getW(), self.output.getW()]

    def getbs(self):
        """
        Return the biases for each layer
        Return a list containing bias for first layer then second and so on...
        """
        return [self.input.getb(), self.output.getb()]

class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, indim, outdim, hiddenlayers=[100,100], alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.input = LinearMap(indim, hiddenlayers[0], alpha=alpha, lr=lr)
        self.relu1 = ReLU(dropout_probability=dropout_probability)
        self.hidden = LinearMap(hiddenlayers[0], hiddenlayers[1], alpha=alpha, lr=lr)
        self.relu2 = ReLU(dropout_probability=dropout_probability)
        self.output = LinearMap(hiddenlayers[1], outdim, alpha=alpha, lr=lr)

    def forward(self, x, train=True):
        x = self.input.forward(x)
        x = self.relu1.forward(x, train=train)
        x = self.hidden.forward(x)
        x = self.relu2.forward(x, train=train)
        x = self.output.forward(x)
        return x

    def backward(self, grad_wrt_out):
        grad_wrt_out = self.output.backward(grad_wrt_out)
        grad_wrt_out = self.relu2.backward(grad_wrt_out)
        grad_wrt_out = self.hidden.backward(grad_wrt_out)
        grad_wrt_out = self.relu1.backward(grad_wrt_out)
        grad_wrt_out = self.input.backward(grad_wrt_out)
        return grad_wrt_out

    def step(self):
        self.input.step()
        self.hidden.step()
        self.output.step()

    def zerograd(self):
        self.input.zerograd()
        self.hidden.zerograd()
        self.output.zerograd()

    def loadparams(self, Ws, bs):
        self.input.loadparams(Ws[0], bs[0])
        self.hidden.loadparams(Ws[1], bs[1])
        self.output.loadparams(Ws[2], bs[2])

    def getWs(self):
        return [self.input.getW(), self.hidden.getW(), self.output.getW()]

    def getbs(self):
        return [self.input.getb(), self.hidden.getb(), self.output.getb()]


def toOneHot(labels, numClasses):
    onehot = np.zeros((labels.shape[0], numClasses), dtype=np.float_)
    onehot[np.arange(labels.shape[0], dtype=np.int_), labels.astype(np.int_)] = 1
    return onehot

def minibatch_SGD(network, loss, trainX, trainY, batch_size=128, shuffle=True):
    train_size = trainX.shape[0]
    if shuffle:
        shuffled_idx = np.random.choice(train_size, train_size, replace=False)
        trainX = trainX[shuffled_idx]
        trainY = trainY[shuffled_idx]
    
    num_batches = train_size // batch_size
    batchedX = trainX[:num_batches*batch_size].reshape(num_batches, -1, trainX.shape[1])
    batchedY = trainY[:num_batches*batch_size].reshape(num_batches, -1, trainY.shape[1])
    losses = np.empty(num_batches)
    accuracy = np.empty(num_batches)
    for batch, (X, Y) in enumerate(zip(batchedX, batchedY)):
        network.zerograd()
        losses[batch] = loss.forward(network.forward(X.T, train=True), Y.T)
        accuracy[batch] = loss.getAccu()
        network.backward(loss.backward())
        network.step()

    return losses.mean(), accuracy.mean()

def eval(network, loss, testX, testY):
    losses = loss.forward(network.forward(testX.T, train=False), testY.T)
    accuracy = loss.getAccu()
    return losses, accuracy

if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle as pk
    import json
    import time

    with open('omniglot_14.pkl','rb') as f:
        data = pk.load(f)
    with open('params.json', 'r') as jsonfile:
        params_dict = json.load(jsonfile)
    
    ((trainX,trainY),(testX,testY)) = data
    trainY = toOneHot(trainY, 14)
    testY = toOneHot(testY, 14)
    indim, outdim = trainX.shape[1], trainY.shape[1]

    # single layer
    print('Training Single Layer MLP')
    single_layer_params = params_dict['SingleLayer']

    for i in range(len(single_layer_params)):
        print(f'model {i}')
        params = single_layer_params[i]
        print(params)

        mlp = SingleLayerMLP(indim, outdim, params['hidden'], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/single_layer_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(single_layer_params)):
        dfs.append(pd.read_csv(f'summary/single_layer_{i}.csv'))

    for i in range(len(single_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(single_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/single_layer_loss.png')
    plt.clf()

    for i in range(len(single_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/single_layer_acc.png')
    plt.clf()

    # two layers
    print('Training Two Layers MLP')
    two_layer_params = params_dict['TwoLayers']

    for i in range(len(two_layer_params)):
        print(f'model {i}')
        params = two_layer_params[i]
        print(params)

        mlp = TwoLayerMLP(indim, outdim, [params['hidden'], params['hidden']], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/two_layers_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(two_layer_params)):
        dfs.append(pd.read_csv(f'summary/two_layers_{i}.csv'))

    for i in range(len(two_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(two_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/two_layers_loss.png')
    plt.clf()

    for i in range(len(two_layer_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/two_layers_acc.png')
    plt.clf()

    # Learning rate
    print('Training Two Layers MLP (learning rate experiment)')
    learning_rate_params = params_dict['LearningRate']

    for i in range(len(learning_rate_params)):
        print(f'model {i}')
        params = learning_rate_params[i]
        print(params)

        mlp = TwoLayerMLP(indim, outdim, [params['hidden'], params['hidden']], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/learning_rate_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(learning_rate_params)):
        dfs.append(pd.read_csv(f'summary/learning_rate_{i}.csv'))

    for i in range(len(learning_rate_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(learning_rate_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/learning_rate_loss.png')
    plt.clf()

    for i in range(len(learning_rate_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/learning_rate_acc.png')
    plt.clf()

    # ---------Custom Experiments---------------
    print('Training Two Layers MLP (dropout experiment)')
    dropout_params = params_dict['Dropout']

    for i in range(len(dropout_params)):
        print(f'model {i}')
        params = dropout_params[i]
        print(params)

        mlp = TwoLayerMLP(indim, outdim, [params['hidden'], params['hidden']], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/dropout_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(dropout_params)):
        dfs.append(pd.read_csv(f'summary/dropout_{i}.csv'))

    for i in range(len(dropout_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(dropout_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/dropout_loss.png')
    plt.clf()

    for i in range(len(dropout_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/dropout_acc.png')
    plt.clf()

    # ----------

    print('Training Two Layers MLP (momentum experiment)')
    momentum_params = params_dict['Momentum']

    for i in range(len(momentum_params)):
        print(f'model {i}')
        params = momentum_params[i]
        print(params)

        mlp = TwoLayerMLP(indim, outdim, [params['hidden'], params['hidden']], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/momentum_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(momentum_params)):
        dfs.append(pd.read_csv(f'summary/momentum_{i}.csv'))

    for i in range(len(momentum_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(momentum_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/momentum_loss.png')
    plt.clf()

    for i in range(len(momentum_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/momentum_acc.png')
    plt.clf()

    # ----------

    print('Training Two Layers MLP (num hidden nodes experiment)')
    num_hidden_params = params_dict['NumHidden']

    for i in range(len(num_hidden_params)):
        print(f'model {i}')
        params = num_hidden_params[i]
        print(params)

        mlp = TwoLayerMLP(indim, outdim, [params['hidden'], params['hidden']], params['momentum'], params['dropout'], params['lr'])
        ce_loss = SoftmaxCrossEntropyLoss()

        start = time.time()
        train_losses, train_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        test_losses, test_accuracy = np.empty(params['epochs']), np.empty(params['epochs'])
        for epoch in range(params['epochs']):
            train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(mlp, ce_loss, trainX, trainY)
            test_losses[epoch], test_accuracy[epoch] = eval(mlp, ce_loss, testX, testY)
            print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
                f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
        print(f'Training model {i} finished. Time elapsed: {time.time()-start:.2f} seconds.')

        df = pd.DataFrame({
            'epoch': list(range(params['epochs'])),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'test_loss': test_losses,
            'test_accuracy': test_accuracy})
        fname = f'summary/num_hidden_{i}.csv'
        df.to_csv(fname, index=False)
        print(f'Loss and accuracy data saved at {fname}')

    dfs = []
    for i in range(len(num_hidden_params)):
        dfs.append(pd.read_csv(f'summary/num_hidden_{i}.csv'))

    for i in range(len(num_hidden_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_loss'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
    for i in range(len(num_hidden_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['test_loss'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/num_hidden_loss.png')
    plt.clf()

    for i in range(len(num_hidden_params)):
        plt.plot(dfs[i]['epoch'], dfs[i]['train_accuracy'], label = f'train_{i}', color=f'C{i}', linestyle='-', alpha=0.7)
        plt.plot(dfs[i]['epoch'], dfs[i]['test_accuracy'], label = f'test_{i}', color=f'C{i}', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/num_hidden_acc.png')
    plt.clf()
