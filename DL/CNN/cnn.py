"""
Fall 2022, 10-617
Assignment-2
Programming - CNN
TA in charge: Athiya Deviyani, Anamika Shekhar, Udaikaran Singh

IMPORTANT:
    DO NOT change any function signatures

Sept 2022
"""

from re import L
import numpy as np
import copy

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    N, C, H, W = X.shape
    X_padded = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)))

    W_out = (W + 2*padding + stride - k_width) // stride
    H_out = (H + 2*padding + stride - k_height) // stride

    w_ = np.kron(
        np.tile(
            (np.arange(k_width).reshape(-1,1) + np.arange(0,stride*W_out,stride).reshape(1,-1)),
            (k_height*C, W_out)
        ),
        np.ones((1,N), np.int_)
    )

    h_ = np.tile(
        np.kron(
            (np.arange(k_height).reshape(-1,1) + np.arange(0,stride*W_out,stride).reshape(1,-1)),
            np.ones((k_width,N*H_out), np.int_)),
        (C,1)
    )

    c_ = np.kron(np.arange(C).reshape(-1,1), np.ones((k_height*k_width, H_out*W_out*N), np.int_))

    n_ = np.tile(np.arange(N).reshape(1,-1), (k_height*k_width*C, H_out*W_out))

    out = X_padded[n_, c_, h_, w_]
    
    return out
    
    

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    N, C, H, W = X_shape

    W_out = (W + 2*padding + stride - k_width) // stride
    H_out = (H + 2*padding + stride - k_height) // stride

    w_ = np.kron(
        np.tile(
            (np.arange(k_width).reshape(-1,1) + np.arange(0,stride*W_out,stride).reshape(1,-1)),
            (k_height*C, W_out)
        ),
        np.ones((1,N), np.int_)
    )

    h_ = np.tile(
        np.kron(
            (np.arange(k_height).reshape(-1,1) + np.arange(0,stride*W_out,stride).reshape(1,-1)),
            np.ones((k_width,N*H_out), np.int_)),
        (C,1)
    )

    c_ = np.kron(np.arange(C).reshape(-1,1), np.ones((k_height*k_width, H_out*W_out*N), np.int_))

    n_ = np.tile(np.arange(N).reshape(1,-1), (k_height*k_width*C, H_out*W_out))
    
    X_grad = np.zeros((N,C,H+2*padding,W+2*padding))

    # debug
    # print(X_grad.shape, n_.shape, grad_X_col.shape)

    np.add.at(X_grad, (n_,c_,h_,w_), grad_X_col)
    
    # remove padding
    if padding:
        X_grad = X_grad[:,:,padding:-padding,padding:-padding]
    
    return X_grad


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
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x):
        self.active = x > 0.
        return np.maximum(x, 0.)

    def backward(self, grad_wrt_out):
        return np.where(self.active, grad_wrt_out, 0.)


class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        # (batch_size, ., ., ...) -> (batch_size, -1)
        self.X_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        return dloss.reshape(self.X_shape)


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.C, self.height, self.width = input_shape
        self.f, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6/((self.f+self.C)*self.k_height*self.k_width))
        self.W = np.random.uniform(-b, b, (self.f, self.C, self.k_height, self.k_width))
        self.b = np.zeros((self.f, 1))
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.momentumW = np.zeros(self.W.shape)
        self.momentumb = np.zeros(self.b.shape)
        

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        we recommend you use im2col here
        """
        self.X_shape = inputs.shape
        self.batch_size = inputs.shape[0]
        self.padding = pad
        self.stride = stride
        self.X_col = im2col(inputs, self.k_height, self.k_width, padding=self.padding, stride=self.stride) # (C*k_height*k_width, H*W*BS)
        self.w_col = self.W.reshape(self.f, -1) # (f, C*k_height*k_width)
        H_out = (self.height + 2*pad + stride - self.k_height) // stride
        W_out = (self.width + 2*pad + stride - self.k_width) // stride
        return np.rollaxis(((self.w_col @ self.X_col) + self.b).reshape((self.f, H_out, W_out, -1)), 3)

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        dout = np.rollaxis(dloss, 0, 4).reshape(self.f, -1) # (f, H*W*BS)
        dX_col = self.w_col.T @ dout
        dX = im2col_bw(dX_col, self.X_shape, self.k_height, self.k_width, padding=self.padding, stride=self.stride)

        dW = (dout @ self.X_col.T).reshape(self.W.shape)
        db = dout.sum(axis=1)[:,None]
        self.dW, self.db = dW, db
        return [dW, db, dX]

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Here we divide gradients by batch_size.
        """
        self.momentumW = momentum_coeff * self.momentumW + self.dW / self.batch_size
        self.momentumb = momentum_coeff * self.momentumb + self.db / self.batch_size
        self.W -= learning_rate * self.momentumW
        self.b -= learning_rate * self.momentumb
        

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return [self.W, self.b]


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_height, self.filter_width = filter_shape
        self.filter_size = self.filter_height * self.filter_width
        self.stride = stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (batch_size, C, H, W)
        """
        self.batch_size, self.C, self.H, self.W = inputs.shape
        W_out = (self.W + self.stride - self.filter_width) // self.stride
        H_out = (self.H + self.stride - self.filter_height) // self.stride
        self.pool_col = im2col(inputs, self.filter_height, self.filter_width, padding=0, stride=self.stride) # (C*k_height*k_width, H_out*W_out*BS)
        self.pool_col = self.pool_col.reshape(self.C, self.filter_size, -1)
        self.pool_max = self.pool_col.max(axis=1) # (C, H_out*W_out*BS)
        return np.rollaxis(self.pool_max.reshape(self.C, H_out, W_out, self.batch_size), 3)

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        dout = np.rollaxis(dloss, 0, 4).reshape(dloss.shape[1], -1) # (f, H*W*BS)
        dout = np.repeat(dout, self.filter_size, axis=0)
        self.which_max = (self.pool_col == self.pool_max[:,None]).reshape(dout.shape)
        dX = im2col_bw(dout*self.which_max, (self.batch_size, self.C, self.H, self.W), self.filter_height, self.filter_width, padding=0, stride=self.stride)
        return dX



class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.indim, self.outdim = indim, outdim
        b = np.sqrt(6)/np.sqrt(indim+outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.momentumW = np.zeros(self.W.shape)
        self.momentumb = np.zeros(self.b.shape)

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.batch_size = inputs.shape[0]
        self.input = inputs
        self.output = self.b.T + inputs @ self.W
        return self.output

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        dX = dloss @ self.W.T
        self.dW = self.input.T @ dloss
        self.db = dloss.sum(axis=0)[:,None]
        return [self.dW, self.db, dX]

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.momentumW = momentum_coeff * self.momentumW + self.dW / self.batch_size
        self.momentumb = momentum_coeff * self.momentumb + self.db / self.batch_size
        self.W -= learning_rate * self.momentumW
        self.b -= learning_rate * self.momentumb

    def get_wb_fc(self):
        """
        Return weights and biases as a tuple
        """
        return (self.W, self.b)


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in  the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        self.softmax_out = np.exp(logits.T)
        self.softmax_out /= np.sum(self.softmax_out, axis=0)
        self.pred = self.softmax_out.argmax(axis=0)
        self.cross_entropy_loss = -(labels.T*np.log(self.softmax_out)).sum()
        self.labels = labels
        if get_predictions:
            return (self.pred, self.cross_entropy_loss)
        return self.cross_entropy_loss


    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        return self.softmax_out.T - self.labels

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        true = self.labels.argmax(axis=0)
        return (self.pred == true).mean()



class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, input_shape=(3,32,32), filter_shape=(1,5,5), padding=2, stride=1, pool_shape=(2,2), pool_stride=2, outdim=20):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """

        C,H,W = input_shape
        f,k_height,k_width = filter_shape
        p_height,p_width = pool_shape
        self.padding = padding
        self.stride = stride
        W_conv = (W + 2*padding + stride - k_width) // stride
        H_conv = (H + 2*padding + stride - k_height) // stride
        W_pool = (W_conv + pool_stride - p_width) // pool_stride
        H_pool = (H_conv + pool_stride - p_height) // pool_stride

        self.conv = Conv(input_shape, filter_shape)
        self.relu = ReLU()
        self.pool = MaxPool(pool_shape, pool_stride)
        self.flatten = Flatten()
        self.fc = LinearLayer(H_pool*W_pool*f, outdim)
        self.softmax = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        x = self.conv.forward(inputs, stride=self.stride, pad=self.padding)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        pred, loss = self.softmax.forward(x, y_labels, get_predictions=True)
        return loss, pred


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.fc.backward(grad)[2]
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad)[2]
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv.update(learning_rate, momentum_coeff)
        self.fc.update(learning_rate, momentum_coeff)
    


class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, input_shape=(3,32,32), filter_shape=(1,5,5), padding=2, stride=1, pool_shape=(2,2), pool_stride=2, outdim=20):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU,LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        
        """
        
        C,H,W = input_shape
        f,k_height,k_width = filter_shape
        p_height,p_width = pool_shape
        self.padding = padding
        self.stride = stride
        W_conv = (W + 2*padding + stride - k_width) // stride
        H_conv = (H + 2*padding + stride - k_height) // stride
        W_pool = (W_conv + pool_stride - p_width) // pool_stride
        H_pool = (H_conv + pool_stride - p_height) // pool_stride
        W_conv2 = (W_pool + 2*padding + stride - k_width) // stride
        H_conv2 = (H_pool + 2*padding + stride - k_height) // stride
        W_pool2 = (W_conv2 + pool_stride - p_width) // pool_stride
        H_pool2 = (H_conv2 + pool_stride - p_height) // pool_stride

        self.conv1 = Conv(input_shape, filter_shape)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(pool_shape, pool_stride)
        self.conv2 = Conv((f, H_pool, W_pool), filter_shape)
        self.relu2 = ReLU()
        self.pool2 = MaxPool(pool_shape, pool_stride)
        self.flatten = Flatten()
        self.fc = LinearLayer(H_pool2*W_pool2*f, outdim)
        self.softmax = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # print(inputs.shape)
        x = self.conv1.forward(inputs, stride=self.stride, pad=self.padding)
        # print(x.shape)
        x = self.relu1.forward(x)
        # print(x.shape)
        x = self.pool1.forward(x)
        # print(x.shape)
        x = self.conv2.forward(x, stride=self.stride, pad=self.padding)
        # print(x.shape)
        x = self.relu2.forward(x)
        # print(x.shape)
        x = self.pool2.forward(x)
        # print(x.shape)
        x = self.flatten.forward(x)
        # print(x.shape)
        x = self.fc.forward(x)
        # print(x.shape)
        pred, loss = self.softmax.forward(x, y_labels, get_predictions=True)
        return loss, pred


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.fc.backward(grad)[2]
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)[2]
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)[2]
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv1.update(learning_rate, momentum_coeff)
        self.conv2.update(learning_rate, momentum_coeff)
        self.fc.update(learning_rate, momentum_coeff)

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Conv -> Relu -> MaxPool ->Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, input_shape=(3,32,32), filter_shape=(7,3,3), padding=2, stride=1, pool_shape=(2,2), pool_stride=2, outdim=20):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        
        C,H,W = input_shape
        f,k_height,k_width = filter_shape
        p_height,p_width = pool_shape
        self.padding = padding
        self.stride = stride
        W_conv = (W + 2*padding + stride - k_width) // stride
        H_conv = (H + 2*padding + stride - k_height) // stride
        W_pool = (W_conv + pool_stride - p_width) // pool_stride
        H_pool = (H_conv + pool_stride - p_height) // pool_stride
        W_conv2 = (W_pool + 2*padding + stride - k_width) // stride
        H_conv2 = (H_pool + 2*padding + stride - k_height) // stride
        W_pool2 = (W_conv2 + pool_stride - p_width) // pool_stride
        H_pool2 = (H_conv2 + pool_stride - p_height) // pool_stride
        W_conv3 = (W_pool2 + 2*padding + stride - k_width) // stride
        H_conv3 = (H_pool2 + 2*padding + stride - k_height) // stride

        self.conv1 = Conv(input_shape, filter_shape)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(pool_shape, pool_stride)
        self.conv2 = Conv((f, H_pool, W_pool), filter_shape)
        self.relu2 = ReLU()
        self.pool2 = MaxPool(pool_shape, pool_stride)
        self.conv3 = Conv((f, H_pool2, W_pool2), filter_shape)
        self.relu3 = ReLU()
        self.flatten = Flatten()
        self.fc = LinearLayer(H_conv3*W_conv3*f, outdim)
        self.softmax = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # print(inputs.shape)
        x = self.conv1.forward(inputs, stride=self.stride, pad=self.padding)
        # print(x.shape)
        x = self.relu1.forward(x)
        # print(x.shape)
        x = self.pool1.forward(x)
        # print(x.shape)
        x = self.conv2.forward(x, stride=self.stride, pad=self.padding)
        # print(x.shape)
        x = self.relu2.forward(x)
        # print(x.shape)
        x = self.pool2.forward(x)
        # print(x.shape)
        x = self.conv3.forward(x, stride=self.stride, pad=self.padding)
        # print(x.shape)
        x = self.relu3.forward(x)
        # print(x.shape)
        x = self.flatten.forward(x)
        # print(x.shape)
        x = self.fc.forward(x)
        # print(x.shape)
        pred, loss = self.softmax.forward(x, y_labels, get_predictions=True)
        return loss, pred


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.fc.backward(grad)[2]
        grad = self.flatten.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)[2]
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)[2]
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)[2]
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv1.update(learning_rate, momentum_coeff)
        self.conv2.update(learning_rate, momentum_coeff)
        self.conv3.update(learning_rate, momentum_coeff)
        self.fc.update(learning_rate, momentum_coeff)

def toOneHot(labels, numClasses):
    onehot = np.zeros((labels.shape[0], numClasses), dtype=np.float_)
    onehot[np.arange(labels.shape[0], dtype=np.int_), labels.astype(np.int_)] = 1
    return onehot

def minibatch_SGD(network, trainX, trainY, trainY_oneHot, batch_size=32, learning_rate=0.01, momentum=0.5, shuffle=True):
    train_size = trainX.shape[0]
    if shuffle:
        shuffled_idx = np.random.choice(train_size, train_size, replace=False)
        trainX = trainX[shuffled_idx]
        trainY = trainY[shuffled_idx]
        trainY_oneHot = trainY_oneHot[shuffled_idx]
    
    num_batches = np.ceil(train_size/batch_size).astype(int)
    losses = np.empty(num_batches)
    correct_count = np.empty(num_batches)
    for i in range(num_batches):
        X = trainX[i*batch_size:(i+1)*batch_size]
        Y = trainY[i*batch_size:(i+1)*batch_size]
        Y_oneHot = trainY_oneHot[i*batch_size:(i+1)*batch_size]
        loss, pred = network.forward(X, Y_oneHot)
        losses[i] = loss
        correct_count[i] = (pred == Y).sum()
        network.backward()
        network.update(learning_rate=learning_rate, momentum_coeff=momentum)

    return losses.sum()/train_size, correct_count.sum()/train_size

def eval(network, testX, testY, testY_oneHot):
    test_size = testX.shape[0]
    loss, pred = network.forward(testX, testY_oneHot)
    accuracy = (pred == testY).mean()
    return loss/test_size, accuracy

# Implement the training as you wish. This part will not be autograded.
#Note: make sure to download the data from the provided link on page 17
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    import pickle
    import time
    import tqdm
    
    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    parser.add_argument('--momentum', type=float, default = 0.5)
    parser.add_argument('--num_epochs', type=int, default = 100)
    parser.add_argument('--conv_layers', type=int, default = 1)
    parser.add_argument('--filters', type=int, default = 1)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    with open("data.pkl", "rb") as f:
        dict = pickle.load(f)
        train, test = dict["train"], dict["test"]
    f.close()

    train_data = train['data']
    test_data = test['data']
    train_labels = train['labels']
    test_labels = test['labels']

    #note: you should one-hot encode the labels

    num_train = len(train_data)
    num_test = len(test_data)
    batch_size = args.batch_size
    train_iter = num_train//batch_size + 1
    test_iter = num_test//batch_size + 1

    num_epochs = args.num_epochs
    f = args.filters
    learning_rate = args.learning_rate
    momentum = args.momentum
    layers = args.conv_layers

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_labels_oneHot = toOneHot(train_labels, 20)
    test_labels_oneHot = toOneHot(test_labels, 20)

    input_shape = train_data.shape[1:]

    if layers == 1:
        model = ConvNet(input_shape=input_shape, filter_shape=(f,5,5), outdim=20)
    elif layers == 2:
        model = ConvNetTwo(input_shape=input_shape, filter_shape=(f,5,5), outdim=20)
    elif layers == 3:
        model = ConvNetThree(input_shape=input_shape, filter_shape=(f,3,3), outdim=20)
    else:
        raise NotImplementedError

    start = time.time()
    train_losses, train_accuracy = np.empty(num_epochs), np.empty(num_epochs)
    test_losses, test_accuracy = np.empty(num_epochs), np.empty(num_epochs)
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_losses[epoch], train_accuracy[epoch] = minibatch_SGD(model, train_data, train_labels, train_labels_oneHot, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
        test_losses[epoch], test_accuracy[epoch] = eval(model, test_data, test_labels, test_labels_oneHot)
        print(f'epoch {epoch}: training loss {train_losses[epoch]:.4f} accuracy {train_accuracy[epoch]*100:.2f}% '\
            f'testing loss {test_losses[epoch]:.4f} accuracy {test_accuracy[epoch]*100:.2f}%')
    print(f'Training model finished. Time elapsed: {time.time()-start:.2f} seconds.')

    # best loss 
    print(f'best train loss: {train_losses.min():.4f} at epoch {train_losses.argmin()}')
    print(f'best test loss: {test_losses.min():.4f} at epoch {test_losses.argmin()}')
    print(f'best train accuracy: {train_accuracy.max()*100:.2f}% at epoch {train_accuracy.argmax()}')
    print(f'best test accuracy: {test_accuracy.max()*100:.2f}% at epoch {test_accuracy.argmax()}')

    epochs = list(range(num_epochs))
    plt.plot(epochs, train_losses, label = 'train')
    plt.plot(epochs, test_losses, label = 'test')
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.title(args.title)
    plt.savefig(f'plots/{args.title}_loss.png')
    plt.clf()

    plt.plot(epochs, train_accuracy, label = 'train')
    plt.plot(epochs, test_accuracy, label = 'test')
    plt.legend()
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.title(args.title)
    plt.savefig(f'plots/{args.title}_accu.png')
    plt.clf()