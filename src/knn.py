import numpy as np
import operator
from utils.dataset import mnist_txt
import time
import matplotlib.pyplot as plt


class knn(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def distance(self, x):
        diff = np.tile(x, (self.data.shape[0], 1)) - self.data
        return ((diff ** 2).sum(axis=1)) ** 0.5
        
    def classify(self, x, k):
        distance = self.distance(x)
        distance = distance.argsort()
        cls_count = {}
        for i in range(k):
            vote_label = self.label[distance[i]]
            cls_count[vote_label] = cls_count.get(vote_label, 0) + 1
            
        sorted_cls = sorted(cls_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_cls[0][0]
        
    
def knn_test():
    train_path = 'G:/altas/meching-learning/dataset/mnist_txt/trainingDigits'
    test_path = 'G:/altas/meching-learning/dataset/mnist_txt/testDigits'
    data_loader = mnist_txt.mnist_text_loader(train_path, test_path)
    classifier = knn(data_loader.get_train_data(), data_loader.get_train_label())
    test_data = data_loader.get_test_data()
    test_label = data_loader.get_test_label()
    
    cost_times = []
    error_rate = []
    k_list = [i for i in range(3, 20, 2)]
    for k in k_list:
        cost_time = 0
        err_count = 0
        for i in range(test_data.shape[0]):
            start = time.clock()
            pred = classifier.classify(test_data[i], k)
            end = time.clock()
            cost_time += (end - start)
            if pred != test_label[i]:
                err_count += 1
                
        cost_times.append(cost_time/test_data.shape[0])
        error_rate.append(err_count/test_data.shape[0])
        
    x = k_list
    plt.figure()
    plt.plot(x, cost_times, label='cost time')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(x, error_rate, label='error rate')
    plt.legend()
    plt.show()