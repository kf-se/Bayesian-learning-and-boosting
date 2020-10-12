import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

def computePriorA(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    i = 0
    while i < Npts:
        li = labels[i]
        prior[li] += np.sum(W[li])
        i += 1

    prior /= np.sum(W)

    return prior

def mlParamsA(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    Nks = np.zeros((Nclasses))

    # Calculating mu, implementing eq. 8 from the lab description.
    # Changed as suggested by eq. 13
    # Calculating sigma, implementing eq. 10 from the lab description.
    # Changed as suggested by eq. 14
    for li in range(Nclasses):
        ix = np.where(labels==li)[0]
        Ws = W[ix,:]
        Xs = X[ix,:]
        u = sum(Xs * Ws) / sum(Ws)
        mu[li] = u

        dXsu = Xs - u
        sigma[li] = np.diag(sum(Ws * np.square(dXsu)) / sum(Ws))

    return mu, sigma

def classifyBayesA(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    i = 0
    while i < Npts:
        x = X[i]
        j = 0
        while j < Nclasses:
            sig = sigma[j]
            sigmaDet = np.linalg.det(sig)
            sigmaInv = np.linalg.inv(sig)
            dxmu = x - mu[j]
            logProb[j][i] = -1/2 * np.log(sigmaDet) - 1/2 * np.dot(np.dot(dxmu, sigmaInv), np.transpose(dxmu)) + np.log(prior[j])

            j += 1
        i += 1
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h
