# 31/40 in total
### -2--for not report the progress and final output of selecting hyperparameter
### -5--for not report the test loss and test accuracy


# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
import time
import itertools

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10

def Z1(w1,b1,digits):
    return np.dot(digits,w1)+b1

##def Z2 No Need

def softmax(w2,b2,hidden_state):
    size = hidden_state.shape[0]
    z = hidden_state.dot(w2)+b2
    exp = np.exp(z-z.max(axis=1)[:,np.newaxis])# Made a mistake here once!
    s = exp.sum(axis=1).reshape(size,1)# Made a mistake here once!
    exp = exp/s
    return exp

def relu(z1):
    z1[z1 < 0] = 0
    return z1
    #return np.clip(z1,0,None)
    #another mistake here !

def Jce (w1,b1,w2,b2,digits, labels, alpha = 0.):
    size = digits.shape[0]
    z1 = Z1(w1,b1,digits)
    hidden_state = relu(z1)
    yhat = softmax(w2, b2, hidden_state)
    yhat = yhat.clip(1e-5,1.-1e-5)
    diag = labels * np.log(yhat)
    J = (-1.0 / size) * diag.sum() + 0.5 * alpha * (w1**2).sum() + 0.5 * alpha * (w2**2).sum()
    return J

def gradRelu(z):
    z[z > 0] = 1
    z[z <= 0] = 0
    return z

def gradJce(w1,b1,w2,b2, digits, labels, alpha = 0.):
    size = digits.shape[0]
    z1 = Z1(w1,b1,digits)
    hidden_state = relu(z1)
    yhat = softmax(w2, b2, hidden_state)
    z1 = np.dot(digits,w1)+b1
    g = (yhat - labels).dot(w2.T) * gradRelu(z1)
    Gw1 = np.dot(digits.T,g) + alpha * w1
    Gw2 = (hidden_state.T).dot(yhat - labels) + alpha *w2
    return Gw1,Gw2

def SGDbatches(trainingdigits,trainingLabels,batch_size):
    total_size = trainingdigits.shape[0] ### -2--should shuffle the data
    for i in np.arange(0,total_size,batch_size):
        yield (trainingdigits[i:i+batch_size],trainingLabels[i:i+batch_size])

def SGDwithCE (trainingdigits, trainingLabels, testingdigits, testingLabels, alpha, lr, batch_size, n_unit, n_epoch):
    w1 = np.random.randn(INPUT_DIM,n_unit)
    w2 = np.random.randn(n_unit,OUTPUT_DIM)
    b1 = np.random.randn(n_unit)
    b2 = np.random.randn(OUTPUT_DIM)
    start = 0
    while(start < n_epoch):
        for x,y in SGDbatches(trainingdigits,trainingLabels,batch_size):
            grad1, grad2 = gradJce(w1,b1,w2,b2,x,y,alpha)
            w1 -= lr * grad1
            w2 -= lr * grad2
        start = start + 1
        print ("Epoch:", '%04d' % (start))
        print ("Tr Loss with CE: %.6f" % Jce(w1,b1,w2,b2, trainingdigits, trainingLabels))
    return w1,b1,w2,b2

def Acc(w1,b1,w2,b2,Digits,Labels):
    z1 = Z1(w1,b1,Digits)
    hidden_state = relu(z1)
    predicted = softmax(w2,b2,hidden_state)
    index = predicted.argmax(axis=1)
    index_true = Labels.argmax(axis=1)
    accuarcy = np.true_divide((index == index_true).sum(),len(index))
    return accuarcy

def reportCostsCE (w1,b1,w2,b2, trainingdigits, trainingLabels, testingdigits, testingLabels, alpha = 0.):
    print ("Training cost: {}".format(Jce(w1,b1,w2,b2, trainingdigits, trainingLabels, alpha)))
    print ("Validation cost:  {}".format(Jce(w1,b1,w2,b2, testingdigits, testingLabels, alpha)))

if __name__ == "__main__":
    # Load data
    if ('trainingdigits' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingImages = np.load("mnist_train_images.npy")
        trainingLabels = np.load("mnist_train_labels.npy")
        testingImages = np.load("mnist_test_images.npy")
        testingLabels = np.load("mnist_test_labels.npy")
        validationImages = np.load("mnist_validation_images.npy")
        validationLabels = np.load("mnist_validation_labels.npy")
        trainingLabels = trainingLabels.astype(int)
        testingLabels = testingLabels.astype(int)
        validationLabels = validationLabels.astype(int)
    params = {
        'n_unit':[30,40,50],
        'lr':[0.001, 0.005, 0.01],
        'batch_size':[16,32,64,128],
        'n_epoch':[10,20,30],
        'alpha':[0.1,0.5,1,]
    }
    preAcc = 0
    for i in itertools.product(params['n_unit'],params['lr'],params['batch_size'],params['n_epoch'],params['alpha']):
        w1,b1,w2,b2 = SGDwithCE(trainingImages, trainingLabels, testingImages, testingLabels,lr = i[1],batch_size = i[2], n_unit = i[0], n_epoch = i[3],alpha=i[4])
        reportCostsCE(w1,b1,w2,b2, trainingImages, trainingLabels, validationImages,validationLabels)
        acc = Acc(w1,b1,w2,b2,validationImages,validationLabels)
        if acc>preAcc:
            preAcc = acc
            best_para = i
            w1_b,b1_b,w2_b,b2_b = w1,b1,w2,b2
        print ("parameters:",i)
        print ("Accuarcy on validation set:",acc)
    print ("best parameters:",best_para)
    print ("accuracy on TestSet:",Accuarcy(w1_b,b1_b,w2_b,b2_b,testingImages,testingLabels))
