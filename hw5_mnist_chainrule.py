import scipy.io as sio
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
data=sio.loadmat('mnist_all.mat')
trainX = np.concatenate((data['train0'],data['train1']),0)
trainY = np.concatenate((np.zeros(len(data['train0'])),np.ones(len(data['train1']))),0)
testX = np.concatenate((data['test0'],data['test1']),0)
testY = np.concatenate((np.zeros(len(data['test0'])),np.ones(len(data['test1']))),0)
permutation = np.random.permutation(trainY.shape[0])
trainX = trainX[permutation, :]
trainY = trainY[permutation]
trainX = preprocessing.scale(trainX.astype(np.float64))

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def forward_1layer(h0,W,b):
    h = np.transpose(W).dot(h0)+b
    return sigmoid(h)

def forward(x, w1, b1, w2, b2, w3, b3):
    f = forward_1layer(forward_1layer(forward_1layer(x,w1,b1),w2,b2),w3,b3)
    return f

def batch_generator(all_data , batch_size, shuffle=True):
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]

H=8

w1 = np.mat(np.random.random((784,H)))
b1 = np.mat(np.random.random((H,1)))
w2 = np.mat(np.random.random((H,H)))
b2 = np.mat(np.random.random((H,1)))
w3 = np.mat(np.random.random((H,1)))
b3 = np.mat(np.random.random((1,1)))

T = 4096
batch_size= 256
LearningRate = 8
batch_gen = batch_generator([trainX, trainY],  batch_size)
acc_100 = []
acc_100t = []
for i in range(T):
    if i%512==0:
        LearningRate = LearningRate/2
    batch_x, batch_y = next(batch_gen)
    d_w3 = np.mat(np.zeros((H, 1)))
    d_b3 = np.mat(np.zeros((1, 1)))
    d_w2 = np.mat(np.zeros((H, H)))
    d_b2 = np.mat(np.zeros((H, 1)))
    d_w1 = np.mat(np.zeros((784, H)))
    d_b1 = np.mat(np.zeros((H, 1)))
    for j in range(batch_size):
        h1 = forward_1layer(np.transpose(np.mat(batch_x[j, :])), w1, b1)
        h2 = forward_1layer(h1, w2, b2)
        fx = forward_1layer(h2, w3, b3)
        sigma3 = fx-batch_y[j]
        d_w3 = d_w3 + h2*sigma3
        d_b3 = d_b3 + sigma3
        sigma2 = np.multiply(np.multiply(w3.dot(sigma3),h2),(1-h2))
        d_w2 = d_w2 + h1.dot(np.transpose(sigma2))
        d_b2 = d_b2 + sigma2
        sigma1 = np.multiply(np.multiply(w2.dot(sigma2), h1), (1-h1))
        a = np.transpose(np.mat(batch_x[j, :])).dot(np.transpose(sigma1))
        d_w1 = d_w1 + np.transpose(np.mat(batch_x[j, :])).dot(np.transpose(sigma1))
        d_b1 = d_b1 + sigma1

    w1 = w1 - LearningRate * d_w1/batch_size
    b1 = b1 - LearningRate * d_b1/batch_size
    w2 = w2 - LearningRate * d_w2/batch_size
    b2 = b2 - LearningRate * d_b2/batch_size
    w3 = w3 - LearningRate * d_w3/batch_size
    b3 = b3 - LearningRate * d_b3/batch_size
    if i%100 == 0:
        acc = 0
        acct = 0
        for k in range(len(trainY)):
            if trainY[k]==(forward(np.transpose(np.mat(trainX[k, :])),w1,b1,w2,b2,w3,b3)>0.5):
                acc = acc + 1
        acc_100 = acc_100 + [acc/len(trainY)*100]
        for f in range(len(testY)):
            if testY[f]==(forward(np.transpose(np.mat(testX[f, :])),w1,b1,w2,b2,w3,b3)>0.5):
                acct = acct + 1
        acc_100t = acc_100t + [acct/len(testY)*100]
        # print(acc/len(trainY)*100,acct/len(testY)*100)
plt.figure()
plt.plot(acc_100, label='training')
plt.plot(acc_100t, label='testing')
plt.text(acc_100.index(max(acc_100)),max(acc_100),max(acc_100))
plt.text(acc_100t.index(max(acc_100t)),max(acc_100t),max(acc_100t))
plt.xlabel("# 100 training batch")
plt.ylabel("Accuracy%")
plt.legend(loc='lower right')
plt.ylim((40,100))
plt.show()