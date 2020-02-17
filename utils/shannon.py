import math


def get_ent(dataset):
    '''
    @brief  计算信息熵
    '''
    label_count = {}
    for vec in dataset:
        label = vec[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        
        label_count[label] += 1
        
    ent = 0.0
    for key in label_count:
        for key in label_count:
            prob = float(label_count[key]) / len(dataset)
            ent -= prob * math.log(prob, 2)
    
    return ent
    
    
def get_gain(dataset, axis):
    '''
    @brief  计算数据集的信息增益
    @param  axis    当前属性
    @param  dataset 每一行数据的最后一个分量为类标
    '''
    ent = get_ent(dataset)
    values = [line[axis] for line in dataset]
    values = set(values)
    gain = ent
    for value in values:
        sub_dataset = [line for line in dataset if line[axis] == value]
        sub_ent = get_ent(sub_dataset)
        gain -= len(sub_dataset)/len(dataset) * sub_ent
        
    return gain
    
    
def get_iv(dataset, axis):
    values = [line[axis] for line in dataset]
    values = set(values)
    iv = 0.0
    for value in values:
        sub_dataset = [line for line in dataset if line[axis] == value]
        iv -= len(sub_dataset)/len(dataset)
        
    return iv
    
    
def get_gain_ratio(dataset, axis):
    '''
    @brief  计算增益率
    '''
    gain = get_gain(dataset, axis)
    iv = get_iv(dataset, axis)
    return gain / iv
        

def get_gini(dataset):
    '''
    @brief  计算基尼值
    '''
    label_count = {}
    for vec in dataset:
        label = vec[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        
        label_count[label] += 1
        
    gini = 1
    for key in label_count:
        prop = label_count[key]/len(dataset)
        gini -= prop * prop
        
    return gini
    

def get_gini_ratio(dataset, axis):
    '''
    @brief  计算基尼系数
    '''
    values = [line[axis] for line in dataset]
    values = set(values)
    gini = 0.0
    for value in values:
        sub_dataset = [line for line in dataset if line[axis] == value]
        prop = len(sub_dataset) / len(dataset)
        gini += prop * get_gini(sub_dataset)
        
    return gini