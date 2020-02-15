'''
手写字符识别
'''
import numpy as np
import os

class mnist_text_loader(object):
    def __init__(self, train_path, test_path):
        self.train = self.load_data(train_path)
        self.test = self.load_data(test_path)
        
    def file2matrix(self, filename):
        with open(filename, 'r') as f:
            ret = np.zeros((1, 1024))
            for i in range(32):
                line = f.readline().strip()
                for j in range(32):
                    ret[0, 32 * i + j] = int(line[j])
                    
        return ret
    
    def load_data(self, path):
        files = os.listdir(path)
        ret = np.zeros((len(files), 1024))
        ret_label = []
        for i in range(len(files)):
            filename = files[i]
            ret[i, :] = self.file2matrix(os.path.join(path, files[i]))
            
            label = int(files[i].split('_')[0])
            ret_label.append(label)
        return ret, ret_label
        
    def get_train_data(self):
        return self.train[0]
        
    def get_test_data(self):
        return self.test[0]
        
    def get_train_label(self):
        return self.train[1]
    
    def get_test_label(self):
        return self.test[1]
        