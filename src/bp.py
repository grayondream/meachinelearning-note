import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import load_data, split_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
acts = {'sigmoid': [sigmoid, dsigmoid]}


class backpropagation(object):
    def __init__(self, layers, act='sigmoid', lr=0.001, regression=True):
        self.layers = layers
        self.lr = lr
        self.caches = {}
        self.grads = {}
        self.act = acts[act][0]
        self.dact = acts[act][1]
        self.parameters = {}
        for i in range(1, len(self.layers)):
            w_key = 'w' + str(i)
            b_key = 'b' + str(i)
            self.parameters[w_key] = np.random.random((self.layers[i], self.layers[i - 1]))
            self.parameters[b_key] = np.zeros((self.layers[i], 1))
        self.regression = regression
        
    def forward(self, x):
        wx = []         #未经过激活函数的值
        f_wx = []       #经过激活函数的值
        
        wx.append(x)
        f_wx.append(x)
        size = len(self.layers)
        for i in range(1, size - 1):
            w_key = 'w' + str(i)
            b_key = 'b' + str(i)
            layer_wx = self.parameters[w_key] @ f_wx[i - 1] + self.parameters[b_key]
            wx.append(layer_wx)
            
            layer_fwx = self.act(wx[-1])
            f_wx.append(layer_fwx)
            
        wx.append(self.parameters['w' + str(size - 1)] @ f_wx[-1] + self.parameters['b' + str(size - 1)])
        if self.regression is True:
            f_wx.append(wx[-1])
        else:
            f_wx.append(self.act(wx[-1]))
        
        self.caches['wx'] = wx
        self.caches['fwx'] = f_wx
        return self.caches, f_wx[-1]
    
    def classify(self, out):
        cls =  np.argmax(out, axis=0)
        return cls
        
    def backward(self, y):
        '''
        @brief  反向传播计算梯度
        '''
        fwx = self.caches['fwx']
        m = y.shape[0]
        size = len(self.layers)    
        cls = fwx[-1]
        if not self.regression:
            cls = self.classify(cls)
        self.grads['dz' + str(size - 1)] = cls - y
        self.grads['dw' + str(size - 1)] = self.grads['dz' + str(size - 1)].dot(fwx[-2].T) / m
        self.grads['db' + str(size - 1)] = np.sum(self.grads['dz' + str(size - 1)], axis=1, keepdims=True) / m
    
        #计算梯度
        for i in reversed(range(1, size - 1)):
            self.grads['dz' + str(i)] = self.parameters['w' + str(i + 1)].T.dot(self.grads["dz" + str(i+1)]) * self.dact(self.caches["wx"][i])
            self.grads["dw" + str(i)] = self.grads['dz' + str(i)].dot(self.caches['fwx'][i-1].T)/m
            self.grads["db" + str(i)] = np.sum(self.grads['dz' + str(i)],axis = 1,keepdims = True) /m
            
        #更新权重和偏置
        for i in range(1, size):
            self.parameters['w' + str(i)] -= self.lr * self.grads['dw' + str(i)]
            self.parameters['b' + str(i)] -= self.lr * self.grads['db' + str(i)]
            
    def loss(self, y):
        if self.regression:
            return np.mean(np.square(self.caches['fwx'][-1]-y))
        else:
            return np.mean(np.square(self.classify(self.caches['fwx'][-1] - y)))
        
    def predict(self, x):
        size = len(self.layers)
        for i in range(1, size - 1):
            w_key = 'w' + str(i)
            b_key = 'b' + str(i)
            x = self.parameters[w_key] @ x + self.parameters[b_key]
            x = self.act(x)
            
        x = self.parameters['w' + str(size - 1)] @ x + self.parameters['b' + str(size - 1)]
        if self.regression is True:
            return x
        else:
            x = self.act(x)
            cls = self.classify(x)
            return cls
    
def regress_test():
    x = np.arange(0.0,1.0,0.01)
    y =20* np.sin(2*np.pi*x)
    plt.scatter(x,y)
    
    x = x.reshape(1, 100)
    y = y.reshape(1, 100)
    
    bp = backpropagation([1, 16, 1], lr=0.1)
    
    for i in range(1, 50000):
        caches, al = bp.forward(x)
        bp.backward(y)
        
        if(i%500 == 0):
           print(bp.loss(y))
    plt.scatter(x, al)
    plt.show()


def shuffle_data(data, label):
    permutation = np.random.permutation(label.shape[0])
    data = data[permutation, :]
    label = label[permutation]
    return data, label
    
def visualize_moon(x, y, bp):
    x_aixs = x[:,0]
    y_aixs = x[:,1]

    neg_x_axis = x_aixs[y==0]
    neg_y_axis = y_aixs[y==0]
    
    pos_x_axis = x_aixs[y==1]
    pos_y_axis = y_aixs[y==1]
    
    plt.figure()
    plt.scatter(neg_x_axis,neg_y_axis,c="b",s=10)
    plt.scatter(pos_x_axis,pos_y_axis,c="r",s=10)
    
    
    #网格搜索决策边界
    xs = np.linspace(-10,20,1000)
    ys = np.linspace(-10,10,1000)
    pts = []
    for x in xs:
        pt = []
        for y in ys:
            pt.append([x, y])
    
        pts.append(pt)
    
    cls = []
    for pt in pts:
        cls.append(bp.classify(bp.predict(np.array(pt).transpose())))
        
    ret_x = []
    ret_y = []
    for i in range(len(cls)):
        for j in range(cls[i].shape[0] - 1):
            if cls[i][j] != cls[i][j + 1]:
                ret_x.append(pts[i][j][0])
                ret_y.append(pts[i][j][1])
                break
                
    plt.plot(ret_x, ret_y)
    plt.show()
    
def moon_test():
    file = 'G:/altas/meching-learning/dataset/halfmoon.txt'
    data, label = load_data(file)
    label[label==-1] = 0
    data, label = shuffle_data(data, label)
    train_data, train_label, test_data, test_label = split_data(data, label)
    bp = backpropagation([2,8,2], lr=0.01, regression=True)
    for i in range(1000000):
        _, out = bp.forward(train_data.transpose())
        bp.backward(train_label)
        if i % 500 == 0:
            print(i, bp.loss(train_label))
        
    prd = bp.predict(test_data.transpose())
    prd = bp.classify(prd)
    print('test error ratio is %f' % (float(np.sum(prd != test_label)) / test_label.shape[0]))
    
    visualize_moon(data, label, bp)
    
def bp_test():
    moon_test()