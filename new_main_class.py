from abc import ABC, abstractmethod
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


class Layer(ABC):
    def __init__(self):
        self.prevIn = []
        self.prevOut = []

    def setPrevIn(self, dataIn):
        self.prevIn = dataIn

    def setPrevOut(self, out):
        self.prevOut = out

    def getPrevIn(self):
        return self.prevIn

    def getPrevOut(self):
        return self.prevOut

    def backward(self, gradIn):
        sg = self.gradient()
        grad = np.zeros((gradIn.shape[0], sg.shape[2]))
        for n in range(
                gradIn.shape[0]):  #compute for each observation in batch
            grad[n, :] = gradIn[n, :] @ sg[n, :, :]
        return grad

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof=1)
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        np.random.seed(0)
        self.weights = np.random.uniform(size=(sizeIn, sizeOut),
                                         low=-0.0001,
                                         high=0.0001)
        self.bias = np.random.uniform(size=(1, sizeOut),
                                      low=-0.0001,
                                      high=0.0001)
        self.sw = 0
        self.rw = 0
        self.sb = 0
        self.rb = 0

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        self.bias = bias

    def getSB(self):
        return self.sb

    def setSB(self, sb):
        self.sb = sb

    def getRB(self):
        return self.rb

    def setRB(self, rb):
        self.rb = rb

    def getSW(self):
        return self.sw

    def setSW(self, sw):
        self.sw = sw

    def getRW(self):
        return self.rw

    def setRW(self, rw):
        self.rw = rw

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = dataIn.dot(self.getWeights()) + self.getBias()
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        X = np.copy(self.getWeights())
        return np.transpose(X)

    def updateWeights(self, gradIn, eta=0.0001):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        self.weights -= eta * dJdW
        self.bias -= eta * dJdb
        return self.weights, self.bias

    def updateWeightsAdam(self,
                          gradIn,
                          eta,
                          t,
                          p_1=0.9,
                          p_2=0.9999,
                          delta=10**-8):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]

        self.sw = p_1 * self.sw + (1 - p_1) * dJdW
        self.rw = p_2 * self.rw + (1 - p_2) * np.power(dJdW, 2)
        sw_hat = self.sw / (1 - np.power(p_1, t))
        rw_hat = self.rw / (1 - np.power(p_2, t))

        self.weights -= eta * (sw_hat / (np.sqrt(rw_hat) + delta))

        self.sb = p_1 * self.sb + (1 - p_1) * dJdb
        self.rb = p_2 * self.rb + (1 - p_2) * np.power(dJdb, 2)
        sb_hat = self.sb / (1 - np.power(p_1, t))
        rb_hat = self.rb / (1 - np.power(p_2, t))
        self.bias -= eta * (sb_hat / (np.sqrt(rb_hat) + delta))
        return self.weights, self.bias, self.sw, self.rw, self.sb, self.rb

    def backward(self, gradIn):
        W = np.copy(self.getWeights())
        fcc_grad = gradIn @ np.transpose(W)
        return fcc_grad


class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.copy(dataIn)
        Y[Y < 0] = 0
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        n = self.getPrevIn().shape[0]

        k = self.getPrevIn().shape[1]
        relu_gradient = np.zeros(shape=(n, k, k))

        for g in range(n):
            X = [0 if i < 0 else 1 for i in self.getPrevIn()[g]]
            np.fill_diagonal(relu_gradient[g], X)

        return (relu_gradient)


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        n = self.getPrevOut().shape[0]
        k = self.getPrevOut().shape[1]
        sigmoid_gradient = np.zeros(shape=(n, k, k))

        for g in range(n):
            X = self.getPrevOut()[g] * (1 - self.getPrevOut()[g]) + 10**-7
            np.fill_diagonal(sigmoid_gradient[g], X)

        return (sigmoid_gradient)


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        out = np.exp(dataIn)
        if out.ndim != 1:
            denom = np.sum(out, axis=1).tolist()
        else:
            denom = np.sum(out)

        for i in range(out.shape[0]):
            if out.ndim != 1:

                out[i] = out[i] / denom[i]
            else:
                out[i] = out[i] / denom

        self.setPrevOut(out)
        return out

    def gradient(self):

        n = self.getPrevOut().shape[0]
        k = self.getPrevOut().shape[1]
        softmax_gradient = np.zeros(shape=(n, k, k))

        for g in range(n):
            gz = np.copy(self.getPrevOut()[g])
            for i in range(len(self.getPrevOut()[g])):
                for j in range(len(self.getPrevOut()[g])):
                    if i == j:
                        softmax_gradient[g][i][j] = gz[i] * (1 - gz[i])
                    else:
                        softmax_gradient[g][i][j] = -gz[i] * gz[j]

        return (softmax_gradient)


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) +
                                                  np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        n = self.getPrevOut().shape[0]
        k = self.getPrevOut().shape[1]
        tanh_gradient = np.zeros(shape=(n, k, k))

        for g in range(n):
            X = 1 - (self.getPrevOut()[g]**2) + 10**-7  #epsilon = 10^(-7)
            np.fill_diagonal(tanh_gradient[g], X)

        return (tanh_gradient)


class LogLoss():
    def eval(self, y, yhat):
        e = 10**-7
        return np.mean(-(y * np.log(yhat + e) +
                         ((1 - y) * np.log(1 - yhat + e))))

    def gradient(self, y, yhat):
        e = 10**-7
        return -((y - yhat) / (yhat * (1 - yhat) + e))


class CrossEntropy():
    def eval(self, y, yhat):
        e = 10**-7

        temp = np.multiply(y, np.log(yhat))
        return np.mean(-(np.sum(temp) + e))

    def gradient(self, y, yhat):
        e = 10**-7
        return -(y / (np.array(yhat) + e))


def model(X, Y, X_val, Y_val, layers, layers_val, eta=0.0001, epochs=15):

    layers = layers

    layers_val = layers_val

    ll_train = []
    ll_val = []
    y_pred_train = []
    y_pred_val = []
    acc_train = []
    acc_val = []

    X = layers[0].forward(X)
    X_val = layers_val[0].forward(X_val)

    Y = one_hot(Y, 10)
    Y_val = one_hot(Y_val, 10)

    for e in tqdm(range(1, epochs + 1)):

        #     forwards!
        h = np.copy(X)
        h_val = np.copy(X_val)
        for i in range(1, len(layers) - 1):
            h = layers[i].forward(h)
            h_val = layers_val[i].forward(h_val)

        y_pred_train = h
        y_pred_val = h_val

        ll_train.append(layers[-1].eval(Y, h))

        #     backwards!
        grad = layers[-1].gradient(Y, h)
        for i in range(len(layers) - 2, 0, -1):
            newgrad = layers[i].backward(grad)

            if isinstance(layers[i], FullyConnectedLayer):

                w, b, sw, rw, sb, rb = layers[i].updateWeightsAdam(
                    grad, eta, e)
                layers_val[i].setWeights(w)
                layers_val[i].setBias(b)
                layers_val[i].setSW(sw)
                layers_val[i].setRW(rw)
                layers_val[i].setSB(sb)
                layers_val[i].setRB(rb)

            grad = newgrad

        ll_val.append(layers_val[-1].eval(Y_val, h_val))

        acc_train.append(accuracy_calc(Y, y_pred_train))
        acc_val.append(accuracy_calc(Y_val, y_pred_val))

    return ll_train, y_pred_train, ll_val, y_pred_val, acc_train, acc_val, epochs, eta


def one_hot(dataIn, classes):
    y_hot = np.zeros((len(dataIn), classes))
    y_hot[np.arange(len(dataIn)), dataIn] = 1
    return y_hot


def mini_batch(feat, lab, batch_size=64):
    batches = []
    for j in range(int(math.ceil(len(feat) / batch_size))):
        f_ind = j * batch_size
        if len(feat) - f_ind >= batch_size:
            batch = batch_size
        else:
            batch = len(feat) % batch_size
        data_train = feat[f_ind:f_ind + batch]
        label_train = lab[f_ind:f_ind + batch]
        batches.append((data_train, label_train))
    return batches


def create_graphs(alls, rows=2, cols=2):
    j = 1
    graph = 1

    for i in range(0, len(alls), 6):

        plt.subplot(rows, cols, j)
        plt.plot(range(0, len(alls[i])),
                 alls[i],
                 color="green",
                 label="Training")

        plt.plot(range(0, len(alls[i + 1])),
                 alls[i + 1],
                 '-.',
                 color="blue",
                 label="Validation")

        plt.title(f"Loss {graph}, epochs = {alls[i+4]}, eta={alls[i+5]}")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(rows, cols, j + 1)
        plt.plot(range(0, len(alls[i + 2])),
                 alls[i + 2],
                 color="green",
                 label="Training")

        plt.plot(range(0, len(alls[i + 3])),
                 alls[i + 3],
                 '-.',
                 color="blue",
                 label="Validation")

        plt.title(f"Accuracy {graph}, epochs = {alls[i+4]}, eta={alls[i+5]}")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        j += 2
        graph += 1

        plt.tight_layout()
    return plt.show()


def accuracy_calc(y, yhat):
    accuracy = []
    for i, j in zip(yhat, y):
        # for x,y in zip(i,j):
        y_train_new = np.zeros(10)
        y_train_new[i.argmax()] = 1

        accuracy.append((np.sum(np.equal(j, y_train_new))) / len(j))

    return sum(accuracy) / len(accuracy) * 100
