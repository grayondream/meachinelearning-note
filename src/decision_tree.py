from utils import shannon
import operator
import matplotlib.pyplot as plt
import pickle

class desicion_tree(object):
    def __init__(self, info_func):
        self.info_func = info_func
        
    def vote_classify(self, cls_list):
        cls_count = {}
        for i in range(k):
            vote_label = self.label[distance[i]]
            cls_count[vote_label] = cls_count.get(vote_label, 0) + 1
            
        sorted_cls = sorted(cls_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_cls[0][0]
        
    def split_dataset(self, data_set, axis, value):
        '''
        @brief  划分数据集
        @param  data_set    数据
        @param  axis        划分依据的特征
        @param  value       需要返回的feaure
        '''
        ret = []
        for vec in data_set:
            if vec[axis] == value:
                ret_vec = vec[:axis]
                ret_vec.extend(vec[axis+1:])
                ret.append(ret_vec)
                
        return ret
    
    def choose_best_feature(self, dataset):
        if self.info_func is shannon.get_ent:
            return self.choose_best_shannon(dataset)
        else:
            best_feature = 0
            max_info = 0
            for i in range(len(dataset[0]) - 1):
                info = self.info_func(dataset, i)
                if info > max_info:
                    max_info = info
                    best_feature = i
                    
            return best_feature
            
    def choose_best_shannon(self, dataset):
        '''
        @brief  选择最好的feature进行划分
        '''
        basic_ent = shannon.get_ent(dataset)
        best_feature = -1
        best_gain = 0.0
        for i in range(len(dataset[0]) - 1):
            feature_list = [ele[i] for ele in dataset]
            feature_list = set(feature_list)
            cur_ent = 0.0
            for value in feature_list:
                sub_dataset = self.split_dataset(dataset, i, value)
                prob = len(sub_dataset)/float(len(dataset))
                cur_ent += prob * shannon.get_ent(sub_dataset)
            
            gain = basic_ent - cur_ent
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                
        return best_feature
    
    def create_tree(self, dataset, labels):
        cls_list = [ele[-1] for ele in dataset]
        if cls_list.count(cls_list[0]) == len(cls_list):
            return cls_list[0]
            
        if len(dataset[0]) == 1:
            return self.vote_classify(cls_list)
            
        best_feature = self.choose_best_feature(dataset)
        best_label = labels[best_feature]
        cur_tree = {best_label:{}}
        
        del labels[best_feature]
        feature_vals = [ele[best_feature] for ele in dataset]
        feature_vals = set(feature_vals)
        for val in feature_vals:
            sub_labels = labels[:]
            cur_tree[best_label][val] = self.create_tree(self.split_dataset(dataset, best_feature, val), sub_labels)
            
        return cur_tree
    
    '''
    决策树分类
    '''
    def classify(self, dec_tree, feature_labels, test_vec):
        root = dec_tree.keys()[0]
        child = dec_tree[root]
        feature_id = feature_labels.index(root)
        key = test_vec[feature_id]
        value = child[key]
        if isinstance(value, dict):
            cls_label = self.classify(value, feature_labels, test_vec)
        else:
            cls_label = value
        
        return cls_label
    
    '''
    存储决策树
    '''
    def save(self, cur_tree, file):
        '''
        @brief  存储树结构
        '''
        with open(file, 'w') as f:
            pickle.dump(cur_tree, f)
            
    def load(self, cur_tree, file):
        with open(file, 'r') as f:
            return pickle.load(f)
        
'''
决策树可视化
'''
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def get_leaf_no(cur_tree):
    leaf_no = 0
    root = list(cur_tree.keys())[0]
    child = cur_tree[root]
    for key in child.keys():
        if type(child[key]).__name__ == 'dict':
            leaf_no += get_leaf_no(child[key])
        else:
            leaf_no += 1
            
    return leaf_no
    
def get_tree_depth(cur_tree):
    max_depth = 0
    root = list(cur_tree.keys())[0]
    child = cur_tree[root]
    for key in child.keys():
        depth = 0
        if type(child[key]).__name__ == 'dict':
            depth += get_leaf_no(child[key])
        else:
            depth = 1
        
        if depth > max_depth:
            max_depth = depth
    return max_depth
    
class decision_tree_viaualization(object):
    def __init__(self):
        self.ax1 = None
        self.x_off = None
        self.y_off = None
        self.total_d = None
        self.total_w = None
        
    def plot_node(self, node_txt, center_pt, parent_pt, node_type):
        self.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va='center', ha='center', bbox=node_type, arrowprops=arrow_args)
    
    def plot_mid_txt(self, ctr_pt, pt_pt, txt_str):
        x_mid = (pt_pt[0] - ctr_pt[0])/2.0 + ctr_pt[0]
        y_mid = (pt_pt[1] - ctr_pt[1])/2.0 + ctr_pt[1]
        self.ax1.text(x_mid, y_mid, txt_str, va='center', ha='center', rotation=30)
    
    def plot_tree(self, cur_tree, pt_pt, node_txt):
        leaf_no = get_leaf_no(cur_tree)
        depth = get_tree_depth(cur_tree)
        root = list(cur_tree.keys())[0]
        ctr_pt = (self.x_off + (1.0 + leaf_no)/2.0/self.total_w, self.y_off)
        
        self.plot_mid_txt(ctr_pt, pt_pt, node_txt)
        self.plot_node(root, ctr_pt, pt_pt, decision_node)
        
        child = cur_tree[root]
        self.y_off = self.y_off - 1.0/self.total_d
        for key in child.keys():
            if type(child[key]).__name__ == 'dict':
                self.plot_tree(child[key], ctr_pt, str(key))
            else:
                self.x_off = self.x_off + 1.0/self.total_w
                self.plot_node(child[key], (self.x_off, self.y_off), ctr_pt, leaf_node)
                self.plot_mid_txt((self.x_off, self.y_off), ctr_pt, str(key))
                
        self.y_off = self.y_off + 1.0/self.total_d
    
    def create_plot(self, cur_tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
        self.total_w = float(get_leaf_no(cur_tree))
        self.total_d = float(get_tree_depth(cur_tree))
        self.x_off = -0.5/self.total_w; 
        self.y_off = 1.0;
        self.plot_tree(cur_tree, (0.5,1.0), '')
        plt.show()
    
    
'''
使用决策树预测隐形眼镜类型
'''
def decision_test():
    file = 'G:/altas/meching-learning/dataset/lenses.txt'
    with open(file, 'r') as f:
        lenses = [line.strip().split('\t') for line in f.readlines()]
        labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        #dec_tree = desicion_tree(shannon.get_ent).create_tree(lenses, labels)
        #dec_tree = desicion_tree(shannon.get_gain).create_tree(lenses, labels)
        dec_tree = desicion_tree(shannon.get_gini_ratio).create_tree(lenses, labels)
        decision_tree_viaualization().create_plot(dec_tree)
        return dec_tree
        
if __name__ == '__main__':
    pass
    