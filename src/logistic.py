import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(file):
    data = []
    label = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = [float(ele) for ele in line]
            line = [1.0] + line
            
            data.append(line[:-1])
            label.append(line[-1])
            
        return data, label
        

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
def grad_ascent(data, label):
    data = np.mat(data)
    label = np.mat(label).transpose()
    m, n = np.shape(data)
    alpha = 0.001
    epochs = 50000
    w = np.ones((n, 1))
    ws = []
    for epoch in range(epochs):
        h = sigmoid(data * w)
        err = label - h
        w = w + alpha * data.transpose() * err
        ws.append(w)
    return w, ws
    

def sto_grad_scent(data, label):
    data = np.array(data)
    m, n = np.shape(data)
    alpha = 0.001
    w = np.ones(n)
    ws = []    
    for i in range(500):
        for epoch in range(m):
            h = sigmoid(np.sum(data[epoch] * w))
            err = label[epoch] - h
            w = w + float(alpha) * float(err) * data[epoch]
            ws.append(w)
    return w, ws
    
def sto_grad_scent_II(data, label):
    data = np.array(data)
    m, n = np.shape(data)
    w = np.ones(n)
    ws = []
    for epoch in range(500):
        ids = list(range(m))
        for i in range(m):
            alpha = 0.0001 + 4 / (1.0 + i + epoch)
            rand_i = int(random.uniform(0, len(ids)))
            h = sigmoid(np.sum(data[rand_i] * w))
            err = label[rand_i] - h
            w = w + float(alpha) * float(err) * np.array(data[rand_i])
            del ids[rand_i]
            ws.append(w)
    return w, ws
    
def visualizae_line(w, data, label):
    '''
    @brief  可视化决策边界
    '''
    data = np.array(data)
    n = np.shape(data)[0]
    x_cord1 = []
    x_cord2 = []
    y_cord1 = []
    y_cord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x_cord1.append(data[i, 1])
            y_cord1.append(data[i, 2])
        else:
            x_cord2.append(data[i, 1])
            y_cord2.append(data[i, 2])
            
    fig = plt.figure()
    plt.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    plt.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-float(w[0]) - float(w[1]) * x) / float(w[2])
    plt.plot(list(x), list(y))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    

def visualizae_w(ws):
    for i in range(3):
        plt.figure()
        plt.plot(list(range(len(ws))), [float(ele[i]) for ele in ws])
        plt.show()
        
def logistic_test():
    file = 'G:/altas/meching-learning/dataset/testset.txt'
    data, label = load_data(file)
    #w, ws = grad_ascent(data, label)
    funcs = [grad_ascent, sto_grad_scent, sto_grad_scent_II]
    for func in funcs:
        w, ws = func(data, label)
        visualizae_line(w, data, label)
        visualizae_w(ws)
            