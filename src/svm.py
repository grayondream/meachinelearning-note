import os
import random
import numpy as np
import matplotlib.pyplot as plt


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
        
    return j
    

def clip_alpha(a_j, h, l):
    '''
    @brief  将a_j的值稳定在[l,h]之间
    '''
    if a_j > h:
        return h
    elif a_j < l:
        return l
    
    return a_j
    
    
def load_data(file):
    data = []
    label = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = [float(ele) for ele in line]
            
            data.append(line[:-1])
            label.append(line[-1])
            
        return data, label
        

def smo_simple(data, label, c, toler, epoches):
    '''
    @brief  smo
    @param  data
    @param  label
    @param  c
    @param  toler   容忍度
    @param  epoches 
    '''
    data, label = np.mat(data), np.mat(label).transpose()
    m,n = np.shape(data)
    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < epoches:
        alpha_pair_changed = 0
        for i in range(m):
            fxi = float((np.multiply(alphas, label).T * (data * data[i,:].T)) + b) #预测值
            ei = fxi - float(label[i])  #误差
            if ((label[i] * ei < -toler) and (alphas[i] < c)) or ((label[i] * ei > toler) and (alphas[i] > 0)):
                j = select_jrand(i, m)  #随机选取另一个alpha
                fxj = float((np.multiply(alphas, label).T * (data * data[j,:].T)) + b) #预测值
                ej = fxj - float(label[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label[i] != label[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                
                if l == h:
                    continue
                    
                eta = 2 * data[i,:] * data[j,:].T - data[i,:] * data[i,:].T - data[j,:] * data[j,:].T
                if eta >=0:
                    continue
                
                alphas[j] -= label[j] * (ei - ej)/eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if (abs(alphas[j] - alpha_j_old) < 0.000001):
                    continue
                
                alphas[i] += label[j] * label[i] * (alpha_j_old - alphas[j])
                
                b1 = b - ei - label[i] * (alphas[i] - alpha_i_old) * data[i,:] * data[i,:].T - label[j] * (alphas[j] - alpha_j_old) * data[i,:] * data[j,:].T
                b2 = b - ei - label[i] * (alphas[i] - alpha_i_old) * data[i,:] * data[j,:].T - label[j] * (alphas[j] - alpha_j_old) * data[i,:] * data[j,:].T
                if 0 < alphas[i] and c > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                    
                alpha_pair_changed += 1
            
        if alpha_pair_changed == 0:
            iter += 1
            #print(iter)    
        else:
            iter = 0
        
    return b, alphas
    

def kernel_trans(x, a, k_tup): 
    '''    
    calc the kernel or transform data to a higher dimensional space
    k_tup 进行模式选择
    '''
    m,n = np.shape(x)
    k = np.mat(np.zeros((m,1)))
    if k_tup[0]=='line': 
        k = x * a.T   #linear kernel
    elif k_tup[0]=='rbf':
        for j in range(m):
            deltaRow = x[j,:] - a
            k[j] = deltaRow * deltaRow.T
        k = np.exp(k/(-1*k_tup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return k
    
class platt_smo(object):
    '''
    @brief  引入核函数
    '''
    def __init__(self, data, label, c, toler, k_tup):
        data = np.mat(data)
        label = np.mat(label)
        
        self.x = data
        self.c = c
        self.label = label.transpose()
        self.tol = toler
        self.m = np.shape(data)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))
        self.k = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.k[:,i] = kernel_trans(self.x, self.x[i,:], k_tup)
            
    def calc_ek(self, k):
        fk = float(np.multiply(self.alphas,self.label).T* self.k[:,k] + self.b)
        ek = fk - float(self.label[k])
        return ek
        
    def select_j_rand(self, i, m):
    	j = i                                 
    	while (j == i):
    		j = int(random.uniform(0, m))
    	return j
 
    def select_j(self, i, ei):
    	max_k = -1; 
    	max_delta_e = 0; 
    	ej = 0 						
    	self.e_cache[i] = [1,ei]  									
    	valid_ecache_list = np.nonzero(self.e_cache[:,0].A)[0]		
    	if (len(valid_ecache_list)) > 1:						
    		for k in valid_ecache_list:   						
    			if k == i: 
    			    continue 							
    			ek = self.calc_ek(k)							
    			delta_e = abs(ei - ek)						
    			if (delta_e > max_delta_e):						
    				max_k = k; 
    				max_delta_e = delta_e; 
    				ej = ek
    		return max_k, ej									
    	else:   												
    		j = self.select_j_rand(i, self.m)							
    		ej = self.calc_ek(j)								
    	return j, ej 			

    def update_ek(self, k):
    	ek = self.calc_ek(k)									
    	self.e_cache[k] = [1,ek]									
     
    def clip_alpha(self, aj,h,l):
    	if aj > h: 
    		aj = h
    	if l > aj:
    		aj = l
    	return aj

    def innerl(self, i):	
    	ei = self.calc_ek(i)
    	if ((self.label[i] * ei < -self.tol) and (self.alphas[i] < self.c)) or ((self.label[i] * ei > self.tol) and (self.alphas[i] > 0)):
    		j,ej = self.select_j(i, ei)
    		alpha_i_old = self.alphas[i].copy(); 
    		alpha_j_old = self.alphas[j].copy();
    		if (self.label[i] != self.label[j]):
    			l = max(0, self.alphas[j] - self.alphas[i])
    			h = min(self.c, self.c + self.alphas[j] - self.alphas[i])
    		else:
    			l = max(0, self.alphas[j] + self.alphas[i] - self.c)
    			h = min(self.c, self.alphas[j] + self.alphas[i])
    		if l == h: 
    			return 0

    		eta = 2.0 * self.k[i,j] - self.k[i,i] - self.k[j,j] #changed for kernel
    		if eta >= 0: 
    			return 0
    			
    		self.alphas[j] -= self.label[j] * (ei - ej)/eta
    		self.alphas[j] = self.clip_alpha(self.alphas[j],h,l)
    		self.update_ek(j)
    		if (abs(self.alphas[j] - alpha_j_old) < 0.00001): 
    			#print("alpha_j变化太小")
    			return 0

    		self.alphas[i] += self.label[j]*self.label[i]*(alpha_j_old - self.alphas[j])
    		self.update_ek(i)
    		b1 = self.b - ei- self.label[i]*(self.alphas[i]-alpha_i_old)*self.k[i,i] - self.label[j]*(self.alphas[j]-alpha_j_old)*self.k[i,j]
    		b2 = self.b - ej- self.label[i]*(self.alphas[i]-alpha_i_old)*self.k[i,j]- self.label[j]*(self.alphas[j]-alpha_j_old)*self.k[j,j]
    		if (0 < self.alphas[i]) and (self.c > self.alphas[i]): 
    		    self.b = b1
    		elif (0 < self.alphas[j]) and (self.c > self.alphas[j]): 
    		    self.b = b2
    		else: 
    		    self.b = (b1 + b2)/2.0
    		
    		return 1
    	else: 
    		return 0
 
    def smo_p(self, epochs):
    	iter = 0 																						
    	entire_set = True; 
    	alpha_pairs_changed = 0
    	while (iter < epochs) and ((alpha_pairs_changed > 0) or (entire_set)):							
    		alpha_pairs_changed = 0
    		if entire_set:																					
    			for i in range(self.m):        
    				alpha_pairs_changed += self.innerl(i)												
    				#print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alpha_pairs_changed))
    			iter += 1
    		else: 																						
    			non_bound_is = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.c))[0]						
    			for i in non_bound_is:
    				alpha_pairs_changed += self.innerl(i)
    				#print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alpha_pairs_changed))
    			iter += 1
    			
    		if entire_set:																				
    			entire_set = False
    		elif (alpha_pairs_changed == 0):																
    			entire_set = True  
    			
    		#print("迭代次数: %d" % iter)
    		
    	return self.b,self.alphas 		


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()
    

def draw_data(data, label):
	data_plus = []                                 
	data_minus = []                               
	for i in range(len(data)):
		if label[i] > 0:
			data_plus.append(data[i])
		else:
			data_minus.append(data[i])
	data_plus_np = np.array(data_plus)            
	data_minus_np = np.array(data_minus)           
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) 
	plt.show()


 
def show_classifer(data, label, w, b, alphas, mode='line'):
	#绘制样本点
	data_plus = []                                
	data_minus = []                                 
	for i in range(len(data)):
		if label[i] > 0:
			data_plus.append(data[i])
		else:
			data_minus.append(data[i])
			
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
	if mode == 'line':
		x1 = max(data)[0]
		x2 = min(data)[0]
		a1, a2 = w
		b = float(b)
		a1 = float(a1[0])
		a2 = float(a2[0])
		y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
		plt.plot([x1, x2], [y1, y2])
	elif mode == 'rbf':
		pass
	
	for i, alpha in enumerate(alphas):
		if abs(alpha) > 0:
			x, y = data[i]
			plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
	plt.show()


def test_rbf(path, k1 = 1.3):
    dataArr,labelArr = load_data(os.path.join(path, 'testSetRBF.txt'))
    clssifier = platt_smo(dataArr, labelArr, 200, 0.0001, ('rbf', k1)) #C=200 important
    b, alphas = clssifier.smo_p(100)
    datMat= np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel_trans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
        
    train_err = errorCount / m
    #dataArr,labelArr = load_data(os.path.join(path, 'testSetRBF2.txt'))
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernel_trans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV, clssifier.alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1    
    #print("the test error rate is: %f" % (float(errorCount)/m))
    test_err = errorCount / m
    return train_err, test_err
	
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
    
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = os.listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(path, k1 = 10):
    dataArr,labelArr = loadImages(os.path.join(path, 'trainingDigits'))
    clssifier = platt_smo(dataArr, labelArr, 200, 0.0001, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArr); 
    labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(classifier.alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel_trans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV, classifier.alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    dataArr,labelArr = loadImages(os.path.join(path, 'testDigits'))
    errorCount = 0
    datMat=np.mat(dataArr); 
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernel_trans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1    
    test_err = error_count / m
    return train_err, test_err
    
def rbf_test():
    file = 'G:/altas/meching-learning/dataset/svm/testset.txt'
    data, label = load_data(file)
    c = 0.6
    epochs = 5
    toler = 0.00001
    #path = 'G:/altas/meching-learning/dataset/svm/'
    path = 'G:/altas/meching-learning/dataset/mnist_txt'
    
    train_err = []
    test_err = []
    x = list(range(1, 10))
    for i in x:
        print(i)
        #t_err, tes_err = test_rbf(path, i)
        t_err, tes_err = testDigits(path, i)
        train_err.append(t_err)
        test_err.append(tes_err)
        
    plt.figure()
    plt.plot(x, train_err, label='train')
    plt.plot(x, test_err, label='test')
    plt.legend()
    plt.show()
    
def svm_test():
    rbf_test()
        