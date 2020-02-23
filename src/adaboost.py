import numpy as np
import matplotlib.pyplot as plt


def load_data(file):
    feat_no = len(open(file).readline().split('\t')) 
    data = []; label = []
    fr = open(file)
    for line in fr.readlines():
        line_arr =[]
        line = line.strip().split('\t')
        for i in range(feat_no-1):
            line_arr.append(float(line[i]))
        data.append(line_arr)
        label.append(float(line[-1]))
    return data, label
    
    
def classify(data, dim, thresh, thresh_op):
    ret = np.ones((np.shape(data)[0], 1))
    if thresh_op == 'lt':
        ret[data[:,dim] <= thresh] = -1.0
    else:
        ret[data[:,dim] > thresh] = -1.0
    return ret
    
def build_stump(data, label, d):
    data = np.mat(data)
    label = np.mat(label).T
    m,n = np.shape(data)
    steps = 10
    best_stump = {}
    bast_cls_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf
    for i in range(n):
        range_min = data[:,i].min()
        randge_max = data[:,i].max()
        step_size = (randge_max - range_min) / steps
        for j in range(-1, steps + 1):
            for op in ['lt', 'gt']:
                thresh = range_min + j * step_size
                pred_val = classify(data, i, thresh, op)
                err = np.mat(np.ones((m, 1)))
                err[pred_val == label] = 0
                weight_err = d.T * err
                if weight_err < min_err:
                    min_err = weight_err
                    bast_cls_est = pred_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh
                    best_stump['op'] = op
    
    return best_stump, min_err, bast_cls_est
    

def ada_train(data, label, epoch=100):
    weak_cls = []
    m = np.shape(data)[0]
    d = np.mat(np.ones((m, 1))/m)
    agg_cls_est = np.mat(np.zeros((m, 1)))
    for i in range(epoch):
        best_stump, err, cls_est = build_stump(data, label, d)
        alpha = 0.5 * float(np.log((1-err)/max(err, 1e-16)))
        best_stump['alpha'] = alpha
        weak_cls.append(best_stump)
        expilsion = np.multiply(-1 * alpha * np.mat(label).T, cls_est)
        d = np.multiply(d, np.exp(expilsion))
        d = d/ d.sum()
        agg_cls_est += alpha * cls_est
        agg_err = np.multiply(np.sign(agg_cls_est) != np.mat(label).T, np.ones((m, 1)))
        err_rate = agg_err.sum()/m
        #print('error is ', err_rate)
        if err_rate == 0.0:
            break
    return weak_cls, agg_cls_est

def ada_classify(data,classifier):
    data = np.mat(data)#do stuff similar to last agg_cls_est in adaBoostTrainDS
    m = np.shape(data)[0]
    agg_cls_est = np.mat(np.zeros((m,1)))
    for i in range(len(classifier)):
        cls_est = classify(data,classifier[i]['dim'],\
                                 classifier[i]['thresh'],\
                                 classifier[i]['op'])#call stump classify
        agg_cls_est += classifier[i]['alpha']*cls_est
    return np.sign(agg_cls_est)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = np.sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if float(classLabels[index]) == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)
    
    
def ada_test():
    train_file = 'G:/altas/meching-learning/dataset/ada/horseColicTraining2.txt'
    test_file = 'G:/altas/meching-learning/dataset/ada/horseColicTest2.txt'
    data, label = load_data(train_file)
    test_data, test_label = load_data(test_file)
    test_label = np.mat(test_label)
    error = []
    x = list(range(20, 21))
    for i in x:
        learner, cls = ada_train(data,label, i)
        pred = ada_classify(test_data, learner)
        pred = pred.reshape(pred.shape[0])
        
        err = pred[pred != test_label]
        err_rate = err.shape[1] / np.shape(pred)[1]
        error.append(err_rate)
        print(i, err_rate)
        plotROC(cls.T, np.mat(label).T)
        
    plt.figure()
    plt.plot(x, error)
    plt.show()