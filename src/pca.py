from numpy import *

def loaddataset(filename, delim='\t'):
    fr = open(filename)
    stringarr = [line.strip().split(delim) for line in fr.readlines()]
    datarr = [map(float,line) for line in stringarr]
    return mat(datarr)

def pca(datamat, topnfeat=9999999):
    meanvals = mean(datamat, axis=0)
    meanremoved = datamat - meanvals #remove mean
    covmat = cov(meanremoved, rowvar=0)
    eigvals,eigvects = linalg.eig(mat(covmat))
    eigvalind = argsort(eigvals)            #sort, sort goes smallest to largest
    eigvalind = eigvalind[:-(topnfeat+1):-1]  #cut off unwanted dimensions
    redeigvects = eigvects[:,eigvalind]       #reorganize eig vects largest to smallest
    lowddatamat = meanremoved * redeigvects#transform data into new dimensions
    reconmat = (lowddatamat * redeigvects.t) + meanvals
    return lowddatamat, reconmat

def replacenanwithmean(): 
    datmat = loaddataset('secom.data', ' ')
    numfeat = shape(datmat)[1]
    for i in range(numfeat):
        meanval = mean(datmat[nonzero(~isnan(datmat[:,i].a))[0],i]) #values that are not nan (a number)
        datmat[nonzero(isnan(datmat[:,i].a))[0],i] = meanval  #set nan values to mean
    return datmat