#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import alternate as alt

# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses,1))

    if W is None:
        W = np.ones((Npts,1))/Npts
        # TODO: compute the values of prior for each class!
        # ==========================
        for i in range(0, Nclasses):
            Nk = len(np.where(labels == i)[0])
            prior[i] = Nk/Npts
        # ==========================
    else:
        assert(W.shape[0] == Npts)
        for i in range(0, Nclasses):
            ik = np.where(labels == i)[0]
            prior[i] = np.sum(W[ik])
        prior /= np.sum(W)
    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#          W - N x 1 matrix of weights
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)     # Nk

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    
    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for c in classes:
        index = np.where(labels == c)[0]
        mu[c] = np.sum(X[index,:]*W[index], axis=0)/np.sum(W[index])

        # Estimate of variance is currently biased, how does this impact the result?
        # To unbias divide by Nclasses - 1
        xd = X[index,:] - mu[c]
        xdd = xd**2 * W[index]
        mean = 1/np.sum(W[index], axis=0) * np.sum(xdd, axis=0)
        sigma[c] =  np.diag(mean)
    # ==========================
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    # log posterior calculated from a discriminative function 
    for c in range(0, Nclasses):
        sigma_d_inv = 1.0/np.diag(sigma[c])
        log_det = np.log(np.prod(np.diag(sigma[c])))
        log_prior = np.log(prior[c])
        for i in range (0, Npts):
            logProb[c][i] = -0.5*log_det - 0.5*np.dot(np.multiply((X[i] - mu[c]), sigma_d_inv), X[i] - mu[c]) + log_prior
    # ==========================
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:
# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.
def assignment1():
    X, labels = genBlobs(centers=5)
    mu, sigma = mlParams(X,labels, W=None)
    prior = computePrior(labels, W=None)
    h = classifyBayes(X, prior, mu, sigma)

    #muA, sigmaA = alt.mlParamsA(X,labels, W=None)
    #priorA = alt.computePriorA(labels, W=None)
    #hA = alt.classifyBayesA(X, priorA, muA, sigmaA)

    #print("prior", prior.shape, "priorA", priorA.shape)
    #print("mu", mu.shape, "muA", muA.shape, "sigma", sigma.shape, "sigmaA", sigmaA.shape)
    #print("h", h, "hA", hA)
    #print("prior", prior, "priorA", priorA)
    #print("mu", mu, "muA", muA, "sigma", sigma, "sigmaA", sigmaA)
    plotGaussian(X,labels,mu,sigma)


    # Call the `testClassifier` and `plotBoundary` functions for this part.
    testClassifier(BayesClassifier(), dataset='iris', split=0.7)
    testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
    plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
    plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)

# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)
    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)
    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        # do classification for each point
        # -1 means counting from the right end, so this would be the last classifier, -2 means 2nd last classifier
        vote = classifiers[-1].classify(X)
        # TODO: Fill in the rest, construct the alphas etc.
        # =========================='
        # Calculate weighted error
        weightsum = np.sum(wCur, axis=0)
        e_t = weightsum
        correctvote = np.where(vote == labels)[0]
        for i in correctvote:
            e_t -= wCur[i]
        # Calculate alpha
        alpha = 0.5*(np.log(1-e_t) - np.log(e_t))
        alphas.append(alpha) # you will need to append the new alpha
        # Update weights
        falsevote = np.where(vote != labels)[0]
        for i in correctvote:
            wCur[i] = wCur[i] * np.exp(-alpha)
        for i in falsevote:
            wCur[i] = wCur[i] * np.exp(alpha)
        # Normalize weights
        wCur /= np.sum(wCur, axis=0)
        # ==========================

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        # NOTE: I do not know how to do this??!!
        for t in range(0, Ncomps):
            h = classifiers[t].classify(X)
            for i in range(0, Npts):
                c = h[i]
                votes[i][c] += alphas[t]#*h[i]
                
        # ==========================
        # one way to compute yPred after accumulating the votes
        ret = np.argmax(votes,axis=1)
        return ret


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.
def assignment_adaboost():
    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)

    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel',split=0.7)

# Now repeat the steps with a decision tree classifier.
#assignment_adaboost()

def assignment6_iris():
    testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
    plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
    plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

#assignment6_iris()
def assignment6_vowel():
    testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)
    plotBoundary(DecisionTreeClassifier(), dataset='vowel',split=0.7)
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
assignment6_vowel()








# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

