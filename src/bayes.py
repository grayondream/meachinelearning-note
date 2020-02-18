import numpy as np
import random


def unique_vocab(dataset):
    '''
    @brief  将字符串列表变成单词表，每个单词仅出现一次
    '''
    ret = set([])
    for line in dataset:
        ret = ret | set(line)
        
    return list(ret)
    

def word2vec(vocab, line):
    '''
    @brief  将line转换为单词表的one-hot表示形式
    '''
    ret = [0] * len(vocab)
    for word in line:
        if word in vocab:
            ret[vocab.index(word)] = 1
        else:
            print("the word %s is not in the vocabulary!" % word)
            
    return ret
    

def bayes_train(dataset, labels):
    doc_no = len(dataset)
    word_no = len(dataset[0])
    pab = np.sum(labels) / float(doc_no)      #类别为1的doc所占的比例
    p0 = np.ones(word_no)
    p1 = np.ones(word_no)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(doc_no):
        if labels[i] == 1:
            p1 += dataset[i]
            p1_denom += np.sum(dataset[i])
        else:
            p0 += dataset[i]
            p0_denom += np.sum(dataset[i])
            
    p1_vec = np.log(p1 / p1_denom)
    p0_vec = np.log(p0 / p0_denom)
    return p0_vec, p1_vec, pab
    

def classify(vec, p0_vec, p1_vec, p_cls):
    p1 = np.sum(vec * p1_vec) + np.log(p_cls)
    p0 = np.sum(vec * p0_vec) + np.log(1 - p_cls)
    if p1 > p0:
        return 1
    else:
        return 0
        
    
'''
垃圾邮件分类
'''
import re
import os
def text_parse(line):
    tokens = re.split(r'\W*', line)
    return [token.lower() for token in tokens if len(token) > 2]
    
def spam_test(path):
    doc_list = []
    cls_list = []
    full_list = []
    for i in range(1, 26):
        spam_file = os.path.join(path, 'spam/%d.txt' % i)
        word_list = text_parse(open(spam_file, 'r').read())
        doc_list.append(word_list)
        full_list.extend(word_list)
        cls_list.append(1)
        
        ham_file = os.path.join(path, 'ham/%d.txt' % i)
        word_list = text_parse(open(ham_file, 'r').read())
        doc_list.append(word_list)
        full_list.extend(word_list)
        cls_list.append(0)
        
    vocab_list = unique_vocab(doc_list)
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        ids = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[ids])
        del train_set[ids]
        
    train_mat = []
    train_cls = []
    for id in train_set:
        train_mat.append(word2vec(vocab_list, doc_list[id]))
        train_cls.append(cls_list[id])
    
    p0, p1, p_spam = bayes_train(np.array(train_mat), np.array(train_cls))
    err_count = 0
    for id in test_set:
        vec = word2vec(vocab_list, doc_list[id])
        if classify(np.array(vec), p0, p1, p_spam) != cls_list[id]:
            err_count += 1
            
    return float(err_count) / len(test_set)
    

def bayes_test():
    path = 'G:/altas/meching-learning/dataset/email'
    ret = 0
    for i in range(100):
        ret += spam_test(path)
    
    print(ret / 100)
    
    