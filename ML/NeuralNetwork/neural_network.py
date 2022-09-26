import numpy as np


def load_data_small():
    """ 
    Load small training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ 
    Load medium training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ 
    Load large training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    Arguments:
        - input: input vector (N, in_features + 1) 
            WITH bias feature added as 1st col
        - p: parameter matrix (out_features, in_features + 1)
            WITH bias parameter added as 1st col (i.e. alpha / beta in the writeup)

    Returns:
        - output vector (N, out_features)
    """
    return np.matmul(input, np.transpose(p))


def sigmoidForward(a):
    """
    Arguments:
        - a: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    return 1/(1+np.exp(-a))


def softmaxForward(b):
    """
    Arguments:
        - b: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    return np.exp(b)/np.sum(np.exp(b))


def crossEntropyForward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K), where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - cross entropy loss (scalar)
    """
    N = hot_y.shape[0]
    return -np.sum(np.multiply(hot_y,np.log(y_hat)))/N


def NNForward(x, y, alpha, beta):
    """
    Arguments:
        - x: input vector (N, M+1)
            WITH bias feature added as 1st col
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col

    Returns (refer to writeup for details):
        - a: 1st linear output (N, D)
        - z: sigmoid output WITH bias feature added as 1st col (N, D+1)
        - b: 2nd linear output (N, K)
        - y_hat: softmax output (N, K)
        - J: cross entropy loss (scalar)

    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    y_hot = np.zeros((y.size, beta.shape[0]))
    y_hot[np.arange(y.size),y] = 1 # transform vector to one-hot matrix
    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    z = np.insert(z,0,1,axis=1) # add first col of 1
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y_hot, y_hat)
    return x,a,z,b,y_hat,J

def softmaxBackward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K) where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    Arguments:
        - prev: previous layer WITH bias feature
        - p: parameter matrix (alpha/beta) WITH bias parameter
        - grad_curr: gradients for current layer

    Returns:
        - grad_param: gradients for parameter matrix (i.e. alpha / beta)
            This should have the same shape as the parameter matrix.
        - grad_prevl: gradients for previous layer

    TIP: Check your dimensions.
    """
    grad_param = np.matmul(np.transpose(grad_curr), prev)
    p_removed_bias = np.delete(p,0,1)
    grad_prevl = np.matmul(grad_curr, p_removed_bias)
    return grad_param, grad_prevl


def sigmoidBackward(curr, grad_curr):
    """
    Arguments:
        - curr: current layer WITH bias feature
        - grad_curr: gradients for current layer

    Returns: 
        - grad_prevl: gradients for previous layer
    """
    curr_removed_bias = np.delete(curr,0,1)
    return np.multiply(np.multiply(grad_curr, curr_removed_bias), (1-curr_removed_bias))


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    Arguments:
        - x: input vector (N, M)
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col
        - z: z as per writeup
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - g_alpha: gradients for alpha
        - g_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    y_hot = np.zeros((y.size, beta.shape[0]))
    y_hot[np.arange(y.size),y] = 1 # transform vector to one-hot matrix
    g_b = softmaxBackward(y_hot, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_x = linearBackward(x, alpha, g_a)
    return g_alpha, g_beta, g_b, g_z, g_a

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N_valid, 1)
        - hidden_units: Number of hidden units
        - num_epoch: Number of epochs
        - init_flag:
            - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
            - False: Initialize weights and bias to 0
        - learning_rate: Learning rate

    Returns:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    M = tr_x.shape[1]
    D = hidden_units
    K = 10

    if init_flag:
        alpha = np.random.uniform(-0.1,0.1,D*M).reshape(D,M)
        alpha = np.insert(alpha,0,0,axis=1)
        beta = np.random.uniform(-0.1,0.1,K*D).reshape(K,D)
        beta = np.insert(beta,0,0,axis=1)
    else:
        alpha = np.zeros((D, M+1))
        beta = np.zeros((K, D+1))

    tr_x_bias = np.insert(tr_x,0,1,axis=1) #(N, M+1)
    valid_x_bias = np.insert(valid_x,0,1,axis=1)
    cross_entropy_train_list = [0 for i in range(num_epoch)]
    cross_entropy_validation_list = [0 for i in range(num_epoch)]

    for n in range(num_epoch):
        for i in range(tr_x.shape[0]):
            x,a,z,b,y_hat,J = NNForward(tr_x_bias[[i]], tr_y[[i]], alpha, beta)
            g_alpha, g_beta, g_b, g_z, g_a = NNBackward(tr_x_bias[[i]], tr_y[[i]], alpha, beta, z, y_hat)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta

        J_train = 0
        for j in range(tr_x.shape[0]):
            y_hat = NNForward(tr_x_bias[[j]], tr_y[[j]], alpha, beta)[4]
            J_train += -np.log(y_hat[0,tr_y[j]])
        J_train = J_train / tr_x.shape[0]

        J_valid = 0
        for j in range(valid_x.shape[0]):
            y_hat = NNForward(valid_x_bias[[j]], valid_y[[j]], alpha, beta)[4]
            J_valid += -np.log(y_hat[0,valid_y[j]])
        J_valid = J_valid / valid_x.shape[0]

        cross_entropy_train_list[n] = J_train
        cross_entropy_validation_list[n] = J_valid

    return alpha, beta, cross_entropy_train_list, cross_entropy_validation_list


def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N-valid, 1)
        - tr_alpha: alpha weights WITH bias
        - tr_beta: beta weights WITH bias

    Returns:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    tr_x_bias = np.insert(tr_x,0,1,axis=1) #(N, M+1)
    valid_x_bias = np.insert(valid_x,0,1,axis=1)
    err_train = 0
    err_valid = 0
    pred_train = [-1 for i in range(tr_x.shape[0])]
    pred_valid = [-1 for i in range(valid_x.shape[0])]
    for i in range(tr_x.shape[0]):
        y_hat = NNForward(tr_x_bias[[i]], tr_y[[i]], tr_alpha, tr_beta)[4]
        y_pred = np.argmax(y_hat)
        pred_train[i] = y_pred
        err_train = err_train if y_pred == tr_y[i] else err_train + 1
    for i in range(valid_x.shape[0]):
        y_hat = NNForward(valid_x_bias[[i]], valid_y[[i]], tr_alpha, tr_beta)[4]
        y_pred = np.argmax(y_hat)
        pred_valid[i] = y_pred
        err_valid = err_valid if y_pred == valid_y[i] else err_valid + 1
    err_train = err_train / tr_x.shape[0]
    err_valid = err_valid / valid_x.shape[0]
    return err_train, err_valid, pred_train, pred_valid


def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.
        
    Arguments:
        - X_train: training input in (N_train, M) array. Each value is binary, in {0,1}.
        - y_train: training labels in (N_train, 1) array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        - X_val: validation input in (N_val, M) array. Each value is binary, in {0,1}.
        - y_val: validation labels in (N_val, 1) array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        - num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        - num_hidden: Positive integer representing the number of hidden units.
        - init_flag: Boolean value of True/False
            - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
            - False: Initialize weights and bias to 0
        - learning_rate: Float value specifying the learning rate for SGD.

    Returns:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    
    alpha, beta, loss_per_epoch_train, loss_per_epoch_val = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epoch, init_rand, learning_rate)
    err_train, err_val, y_hat_train, y_hat_val = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    return (loss_per_epoch_train,
            loss_per_epoch_val,
            err_train,
            err_val,
            y_hat_train,
            y_hat_val)
