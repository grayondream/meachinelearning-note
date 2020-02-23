import matplotlib.pyplot as plt
import numpy as np
import os
import random

def load_data(file):      #general function to parse tab -delimited floats
    feat_no = len(open(file).readline().split('\t')) - 1 #get number of fields 
    data = []
    label = []
    fr = open(file)
    for line in fr.readlines():
        ine_arr =[]
        line = line.strip().split('\t')
        for i in range(feat_no):
            ine_arr.append(float(line[i]))
        data.append(ine_arr)
        label.append(float(line[-1]))
    return data,label

def std_regress(xdata,ydata):
    xmat = np.mat(xdata); ymat = np.mat(ydata).T
    xtx = xmat.T*xmat
    if np.linalg.det(xtx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xtx.I * (xmat.T*ymat)
    return ws
    
    
def std_regress_test():
    path = 'G:/altas/meching-learning/dataset/regress'
    file = os.path.join(path, 'ex0.txt')
    x, y = load_data(file)
    x = np.mat(x)
    y = np.mat(y)
    w = std_regress(x, y)
    print(w)
    fx = x * w
    #plt.plot(x[:,1], y)
    plt.scatter(x[:,1].flatten().A[0],y.T[:,0].flatten().A[0])
    plt.plot(x[:,1], fx)
    plt.show()
    

'''
局部加权回归
'''
#局部加权线性回归
def lwlr(test_data,xdata,ydata,k=1.0):
    xmat = np.mat(xdata); ymat = np.mat(ydata).T
    m = np.shape(xmat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diff = test_data - xmat[j,:]     #
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    xtx = xmat.T * (weights * xmat)
    if np.linalg.det(xtx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xtx.I * (xmat.T * (weights * ymat))
    return test_data * ws

def lwlr_test(test_arr,xdata,ydata,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(test_arr)[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = lwlr(test_arr[i],xdata,ydata,k)
    return y

def lwlr_test_main():
    path = 'G:/altas/meching-learning/dataset/regress'
    file = os.path.join(path, 'ex0.txt')
    x, y = load_data(file)
    xmat = np.mat(x)
    ymat = np.mat(y)
    k = 1
    while k > 0.0001:
        y=lwlr_test(xmat,xmat,ymat,k)
        strInd=xmat[:,1].argsort(0)
        xsort=xmat[strInd][:,0,:]
        fig=plt.figure()
        plt.scatter(xmat[:,1].flatten().A[0],ymat.T[:,0].flatten().A[0])
        plt.plot(xsort[:,1],y[strInd], color='red', label='k%f'%k)
        plt.legend()
        plt.show()
        k = k/10

'''
岭回归
'''
def ridge_regres(xmat,ymat,lam=0.2):
    xtx = xmat.T*xmat
    denom = xtx + np.eye(np.shape(xmat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print( "This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xmat.T*ymat)
    return ws

def rss_error(y_arr,y_pred):
    return ((y_arr-y_pred)**2).sum()

def ridge_test(xarr,y_arr):
    xmat = np.mat(xarr)
    ymat= np.mat(y_arr).T
    ymean = np.mean(ymat,0)
    ymat = ymat - ymean     #to eliminate X0 take mean off of Y
    #regularize X's
    x_mean = np.mean(xmat,0)   #calc mean then subtract it off
    xvar = np.var(xmat,0)      #calc variance of Xi then divide by it
    xmat = (xmat - x_mean)/xvar
    test_pts_no = 30
    wmat = np.zeros((test_pts_no,np.shape(xmat)[1]))
    for i in range(test_pts_no):
        ws = ridge_regres(xmat,ymat,np.exp(i-10))
        wmat[i,:]=ws.T
    return wmat
    
def cross_validation(xarr, y_arr, numFold = 10):
    m = len(y_arr)
    index_list = list(range(m))
    err_mat = np.zeros((numFold, 30))# 每一行都有30个λ得到的结果
    for i in range(numFold):
        trainx = []
        trainy = []
        testx = []
        testy = []
        random.shuffle(index_list)# 把index_list打乱获得随机的选择效果
        for j in range(m):# 划分测试集和训练集
            if j < 0.9*m:
                trainx.append(xarr[index_list[j]])
                trainy.append(y_arr[index_list[j]])
            else:
                testx.append(xarr[index_list[j]])
                testy.append(y_arr[index_list[j]])
        # 30组系数，返回维度为30*8的数组
        wmat = ridge_test(trainx, trainy)

        # 对每一组系数求误差
        # 训练数据做了怎么的处理，新的查询数据就要做怎样的处理才能带入模型
        for k in range(30):
            mattestx = np.mat(testx)
            mattrainx = np.mat(trainx)
            meantrainx = np.mean(trainx, 0) 
            vartrainx = np.var(mattrainx, 0)
            meantrainy = np.mean(trainy, 0)
            mattestx = (mattestx - meantrainx)/vartrainx
            yest = mattestx * np.mat(wmat[k, :]).T + np.mean(trainy)            
            err_mat[i, k] = rss_error(yest.T.A, np.array(testy))
    #print(err_mat)
    meanErrors = np.mean(err_mat, 0) # 每个λ对应的平均误差
    minErrors = float(min(meanErrors))
    # 找到最优的λ之后，选取最后一轮迭代的wmat的对应的λ的权重作为最佳权重
    bestw = wmat[np.nonzero(meanErrors == minErrors)]
    xmat = np.mat(xarr)
    ymat = np.mat(y_arr)
    meanx = np.mean(xmat, 0)
    varx = np.var(xmat, 0)
    # 为了和标准线性回归比较需要对数据进行还原
    print(varx)
    unreg = bestw/varx
    print(unreg)
    print(-1*sum(np.multiply(meanx,unreg)) + np.mean(ymat))


def ridege_test_main():
    path = 'G:/altas/meching-learning/dataset/regress'
    file = os.path.join(path, 'abalone.txt')
    x, y = load_data(file)
    ws=ridge_test(x,y)
    fig=plt.figure()
    plt.plot(ws)
    plt.show()
    cross_validation(x, y)

'''
前向逐步回归
'''
def regularize(xmat):#regularize by columns
    inmat = xmat.copy()
    inmean = np.mean(inmat,0)   #calc mean then subtract it off
    invar = np.var(inmat,0)      #calc variance of Xi then divide by it
    inmat = (inmat - inmean)/invar
    return inmat
    
def stage_wise(xarr,yarr,eps=0.01,epoch=400):#eps步长
    xmat = np.mat(xarr); ymat=np.mat(yarr).T
    ymean = np.mean(ymat,0)
    ymat = ymat - ymean     #can also regularize ys but will get smaller coef
    xmat = regularize(xmat)
    m,n=np.shape(xmat)
    retmat = np.zeros((epoch,n)) #testing code remove
    ws = np.zeros((n,1)); wstest = ws.copy(); wsmax = ws.copy()
    for i in range(epoch):
        lowesterror = np.inf; 
        for j in range(n):#分别计算增加或减少该特征对误差影响
            for sign in [-1,1]:
                wstest = ws.copy()
                wstest[j] += eps*sign
                ytest = xmat*wstest
                rsse = rss_error(ymat.A,ytest.A)
                if rsse < lowesterror:
                    lowesterror = rsse
                    wsmax = wstest
        ws = wsmax.copy()
        retmat[i,:]=ws.T
    return retmat


def stage_test():
    path = 'G:/altas/meching-learning/dataset/regress'
    file = os.path.join(path, 'abalone.txt')
    x, y = load_data(file)
    ws=stage_wise(x, y)
    fig=plt.figure()
    plt.plot(ws)
    plt.show()

def regress_test():
    # std_regress_test()
    #lwlr_test_main()
    #ridege_test_main()
    stage_test()