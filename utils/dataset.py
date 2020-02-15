def load_data(filename):
    with open(filename, 'r') as f:
        data, label = [], []
        for line in f.readlines():
            line = line.split(' ')
            line_data = [int(value) for value in line[:-1]]
            data.append(line_data)
            label.append(int(line[-1]))
            
        return data, label
        

def split_data(data, label, props):
    '''
    @brief  分割训练集和测试集
    @param  data    数据
    @param  label   标签
    @param  props   训练集和测试集比例
    '''
    train_data, test_data, train_label, test_label = [], [], [], []
    
    train_prop = props[0] / sum(props)
    for i in range(len(data)):
        if i / len(data) < train_prop:
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
        
    return train_data, train_label, test_data, test_label