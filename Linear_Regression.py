import numpy as np
import pandas as pd
import argparse
import csv


def PredictedValue(X, W):
    f_x = np.dot(X, W)
    return f_x

def FindGradient(X, Y, f_x):
    gradient = (Y - f_x) * X
    gradient = np.sum(gradient, axis=0)
    return gradient

def FindWeights(W, gradient, eta):
    return W + np.array(eta * gradient).reshape(W.shape)

def SSE(Y, f_x):    
    sse = np.sum(np.square(f_x - Y))    
    return sse

def main():
    args = parser.parse_args()
    file, eta, threshold = args.data, float(args.eta), float(args.threshold)
   
    with open(file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1.0] + row[:-1])
            Y.append([row[-1]])
       
    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(X.shape[1]).astype(float)
    W = W.reshape(X.shape[1], 1)

    iteration = 0
    f_x = PredictedValue(X, W)
    ssePrev = SSE(Y, f_x)
   
    print(iteration,",", end='')
    for w in W.T[0]:
        print(w,",", end='')
    print(ssePrev)
   
    gradient = FindGradient(X, Y, f_x)
    W = FindWeights(W, eta, gradient)
    f_x = PredictedValue(X, W)
    sseCurr = SSE(Y, f_x)
   
    while(abs(sseCurr - ssePrev) > threshold):
       
        iteration += 1
        ssePrev = sseCurr
       
        print(iteration,",", end='')
        for w in W.T[0]:
            print(w,",", end='')
        print(sseCurr)
       
        gradient = FindGradient(X, Y, f_x)
        W = FindWeights(W, eta, gradient)
        f_x = PredictedValue(X, W)
        sseCurr = SSE(Y, f_x)

    print(iteration+1,",", end='')
    for w in W.T[0]:
        print(w,",", end='')
    print(sseCurr)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--eta", help="Learning Rate")    
    parser.add_argument("-t", "--threshold", help="Threshold")    
    main()
   
   

