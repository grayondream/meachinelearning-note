import numpy as np
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from src.decision_tree import decision_tree_viaualization


class node(object):
    '''
    @brief  cart决策树结点
    @param  feat    待切分的特征
    @param  val     待切分的特征值
    @param  left
    @param  right
    '''
    def __init__(self, feat, val, right, left):
        self.feat = feat
        self.val = val
        self.left = left
        self.right = right
        
def reg_leaf(dataset):
    return np.mean(dataset[:,-1])
    
def reg_err(dataset):
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]
    
def bin_split_data(dataset, feature, value):
    rst = dataset[np.nonzero(dataset[:,feature] > value)[0], :]
    snd = dataset[np.nonzero(dataset[:,feature] <= value)[0], :]
    return rst, snd
    

def create_tree(dataset, leaftype=reg_leaf, err_type=reg_err, ops=(1,4)):
    feat, val = choose_best_split(dataset, leaftype, err_type, ops)
    if feat is None:
        return val
    
    ret_tree = {}
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    
    l_set, r_set = bin_split_data(dataset, feat, val)
    ret_tree['left'] = create_tree(l_set, leaftype, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaftype, err_type, ops)
    return ret_tree


def choose_best_split(dataset, leaftype=reg_leaf, err_type=reg_err, ops=(1,4)):
    tol_s, tol_n = ops
    if len(set(dataset[:,-1].T.tolist()[0])) == 1:
        #所有的元素相同
        return None,leaftype(dataset)
        
    m,n = np.shape(dataset)
    s = err_type(dataset)
    best_s = np.inf
    best_id = 0
    best_val = 0
    for id in range(n - 1):
        for val in set((dataset[:, id].T.A.tolist())[0]):
            rst, snd = bin_split_data(dataset, id, val)
            if (np.shape(rst)[0] < tol_n) or (np.shape(snd)[0] < tol_n):
                continue
                
            new_s = err_type(rst) + err_type(snd)
            if new_s < best_s:
                best_id = id
                best_val = val
                best_s = new_s
    
    if (s - best_s) < tol_s:
        return None, leaftype(dataset)
        
    rst, snd = bin_split_data(dataset, best_id, best_val)
    if (np.shape(rst)[0] < tol_n) or (np.shape(snd)[0] < tol_n):
        return None, leaftype(dataset)
        
    return best_id, best_val
    
    
'''
树的剪枝
'''
def is_tree(obj):
    return (type(obj).__name__ == 'dict')
    
def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
        
    return (tree['right'] + tree['left']) / 2.0
    
def prune(tree, dataset):
    '''
    @brief  后剪枝
    '''
    if np.shape(dataset)[0] == 0:
        return get_mean(tree)
        
    l_set, r_set = None, None
    if (is_tree(tree['right'])) or is_tree(tree['left']):
        l_set, r_set = bin_split_data(dataset, tree['sp_ind'], tree['sp_val'])
        
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
        
    if not is_tree(tree['right']) and not is_tree(tree['left']):
        l_set, r_set = bin_split_data(dataset, tree['sp_ind'], tree['sp_val'])
        err_nomerge = np.sum(np.pow(l_set[:-1] - tree['left'], 2)) + np.sum(np.pow(r_set[:-1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right'])/2.0
        err_merge = np.sum(np.power(dataset[:,-1] - tree_mean, 2))
        if err_merge < err_nomerge:
            print('merging')
            return tree_mean
        else:
            return tree
    else:
        return tree
    
    
def load_data(file):
    with open(file, 'r') as f:
        ret = []
        for line in f.readlines():
            line = line.strip().split('\t')
            line = list(map(float, line))
            ret.append(line)
            
        return ret
        
        
'''
GUI
'''
def reg_tree_eval(model, inDat):
    if model is None:
        return 0
        
    return np.float(model)

def model_tree_eval(model, inDat):
    n = shape(inDat)[1]
    X = np.mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return np.float(X*model)
    
def linear_solve(dataSet):   #helper function used in two places
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def model_leaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linear_solve(dataSet)
    return ws

def model_err(dataSet):
    ws,X,Y = linear_solve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat,2))

def treeForeCast(tree, inData, model_eval=reg_tree_eval):
    if not is_tree(tree): return model_eval(tree, inData)
    if inData[tree['sp_ind']] > tree['sp_val']:
        if is_tree(tree['left']): 
            return treeForeCast(tree['left'], inData, model_eval)
        else: return model_eval(tree['left'], inData)
    else:
        if is_tree(tree['right']): 
            return treeForeCast(tree['right'], inData, model_eval)
        else: return model_eval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=reg_tree_eval):
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
    
class cart_gui(object):
    def __init__(self, file):
        self.root = Tk()
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, columnspan=3)
        
        Label(self.root, text="tol_n").grid(row=1, column=0)
        self.tol_nentry = Entry(self.root)
        self.tol_nentry.grid(row=1, column=1)
        self.tol_nentry.insert(0,'10')
        
        Label(self.root, text="tol_s").grid(row=2, column=0)
        self.tol_sentry = Entry(self.root)
        self.tol_sentry.grid(row=2, column=1)
        self.tol_sentry.insert(0,'1.0')
        
        Button(self.root, text="ReDraw", command=self.draw_new_tree).grid(row=1, column=2, rowspan=3)
        self.chk_btn_var = IntVar()
        self.chk_btn = Checkbutton(self.root, text="Model Tree", variable = self.chk_btn_var)
        self.chk_btn.grid(row=3, column=0, columnspan=2)
        
        self.raw_dat = np.mat(load_data(file))
        self.test_dat = np.arange((np.min(self.raw_dat[:,0])), (np.max(self.raw_dat[:,0])),0.01)
        self.re_draw(1.0, 10)
                   
        self.root.mainloop()
    
    def re_draw(self, tol_s, tol_n):
        self.fig.clf()
        self.a = self.fig.add_subplot(111)
        y_hat = 0
        if self.chk_btn_var.get():
            if tol_n < 2:
                tol_n = 2
            
            cur_tree = create_tree(self.raw_dat, model_leaf, model_err, (tol_s, tol_n))
            y_hat = createForeCast(cur_tree, self.test_dat)
            self.tree = cur_tree
        else:
            cur_tree = create_tree(self.raw_dat, ops=(tol_s,tol_n))
            y_hat = createForeCast(cur_tree, self.test_dat)
            self.tree = cur_tree
            
        self.a.scatter(np.array(self.raw_dat[:,0]), np.array(self.raw_dat[:,1]), s=5)
        self.a.plot(self.test_dat, y_hat, linewidth=2.0)
        self.canvas.draw()
        
    def get_inputs(self):
        try: 
            tol_n = int(self.tol_nentry.get())
        except: 
            tol_n = 10 
            print("enter Integer for tolN")
            self.tol_nentry.delete(0, END)
            self.tol_nentry.insert(0,'10')
        try: tol_s = np.float(self.tol_sentry.get())
        except: 
            tol_s = 1.0 
            print("enter Float for tolS")
            self.tol_sentry.delete(0, END)
            self.tol_sentry.insert(0,'1.0')
        return tol_n,tol_s
    
    def draw_new_tree(self):
        tol_n,tol_s = self.get_inputs()#get values from Entry boxes
        self.re_draw(tol_s, tol_n)
    
    def get_tree(self):
        return self.tree
    
def cart_test():
    file = 'G:/altas/meching-learning/dataset/sine.txt'
    gui = cart_gui(file)
    tree = gui.get_tree()
    decision_tree_viaualization().create_plot(tree)
    
if __name__ == '__main__':
    pass
    