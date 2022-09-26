import numpy as np
import matplotlib.pyplot as plt


def NB_XGivenY(XTrain, yTrain):
    """
    Arguments:
        - XTrain: (n, V) array of training documents, where XTrain[i, j] = 1 
            if the i-th document contains word j, 0 otherwise
        - yTrain: (n, 1) array of labels corresponding to each training document

    Returns:
        - D: (2, V) array of MAP estimates, 
            where D[y-1, w-1] is the MAP estimate of P(X_w = 1 | Y = y)
    """
    # Beta(1.001, 1.9) prior
    y1_count = yTrain[yTrain == 1].shape[0]
    y2_count = yTrain[yTrain == 2].shape[0]
    x_y1_count = XTrain[np.concatenate(yTrain == 1)].sum(axis=0)
    x_y2_count = XTrain[np.concatenate(yTrain == 2)].sum(axis=0)
    map_est = np.stack(((x_y1_count+0.001)/(y1_count+0.001+0.9), (x_y2_count+0.001)/(y2_count+0.001+0.9)))
    return np.clip(map_est, 1e-5, 1-1e-5)



def NB_YPrior(yTrain):
    """
    Arguments:
        - yTrain: (n, 1) array of labels corresponding to each training document

    Returns:
        - p: the MLE for P(Y = 1)
    """
    return yTrain[yTrain == 1].shape[0] / yTrain.shape[0]


def NB_Classify(D, p, X):
    """
    Arguments:
        - D: (2, V) array of MAP estimates, 
            where D[y-1, w-1] is the MAP estimate of P(X_w = 1 | Y = y)
        - p: the MLE for P(Y = 1)
        - X: (m, V) array of documents, where X[i, j] = 1 
            if the i-th document contains word j, 0 otherwise

    Returns:
        - yHat: (m, 1) array of predicted labels for each document
    """
    
    func1 = lambda m, v: np.log(D[0,v] if X[m,v] == 1 else 1 - D[0,v])
    func2 = lambda m, v: np.log(D[1,v] if X[m,v] == 1 else 1 - D[1,v])
    y_1_mat = np.fromfunction(np.vectorize(func1), X.shape, dtype=int)
    y_2_mat = np.fromfunction(np.vectorize(func2), X.shape, dtype=int)
    y_1_prob = y_1_mat.sum(axis=1) + np.log(p)
    y_2_prob = y_2_mat.sum(axis=1) + np.log(1-p)
    return_func = lambda x,y: 1 if y_1_prob[x] >= y_2_prob[x] else 2
    return np.fromfunction(np.vectorize(return_func), (X.shape[0], 1), dtype=int)


def ClassificationError(yHat, yTruth):
    """
    Arguments:
        - yHat: (m, 1) array of predicted labels for each document
        - yTruth: (m, 1) array of ground truth labels for each document
    """
    
    return yHat[yHat != yTruth].shape[0] / yHat.shape[0]



if __name__ == '__main__':
    import pickle
    with open('hw2data.pkl', 'rb') as f:
        data = pickle.load(f)

    XTrain, yTrain = data['XTrain'], data['yTrain']
    XTest, yTest = data['XTest'], data['yTest']

    # You may want to convert XTrain / XTest from a sparse matrix to a dense matrix
    XTrain, XTest = XTrain.todense(), XTest.todense()
    
    # Test different training size and their training/testing errors
    ms = np.arange(100,610,30)
    m_count = ms.shape[0]
    XTrains = [XTrain[:m,] for m in ms]
    yTrains = [yTrain[:m] for m in ms]

    trainErrors = np.zeros(m_count)
    testErrors = np.zeros(m_count)

    for i in range(m_count):
        D = NB_XGivenY(XTrains[i], yTrains[i])
        p = NB_YPrior(yTrains[i])

        yHatTrain = NB_Classify(D, p, XTrains[i])
        yHatTest = NB_Classify(D, p, XTest)

        trainErrors[i] = ClassificationError(yHatTrain, yTrains[i])
        testErrors[i] = ClassificationError(yHatTest, yTest)
        print("done {} out of {}".format(i+1, m_count))

    plt.plot(ms, trainErrors)
    plt.plot(ms, testErrors)
    plt.xlabel('m')
    plt.ylabel('error')
    plt.legend(['Training error', 'Test error'])
    plt.savefig('error.png')
    plt.show()

