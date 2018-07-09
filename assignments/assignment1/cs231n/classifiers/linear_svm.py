import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #print("X shape is:",X.shape)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #print("X[i]  is:",X[i])
    #print("scores shape is:",scores.shape)
    #print("scores  is:",scores)
    correct_class_score = scores[y[i]]
    #print("correct_class_score  is:",correct_class_score)
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #print("margin  is:",margin)
      if margin > 0:
        dW[:, y[i]] += -X[i,:].T
        dW[:, j] += X[i,:].T
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #print("dW  is:",dW)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  #print("y  is:",y)
  #print("X shape is:",X.shape)
  #print("W shape is:",W.shape)
  #print("y shape is:",y.shape)
  #print("scores shape is:",scores.shape)
  #print("scores  is:",scores)
  num_classes = W.shape[1]
  num_train = X.shape[0] 
  #print("np.arange(num_train) is",np.arange(num_train))
  scores_correct = scores[np.arange(num_train),y]  
  #print("scores_correct  is:",scores_correct)
  #print("scores_correct shape is:",scores_correct.shape)  
  scores_correct = np.reshape(scores_correct,(num_train,-1))
  #print("scores_correct shape is:",scores_correct.shape)
  #print("scores_correct  is:",scores_correct)
  margins = scores - scores_correct + 1
  #print("margins shape is:",margins.shape)
  #print("margins  is:",margins)
  margins = np.maximum(0,margins)
  #print("margins  is:",margins)
  margins [np.arange(num_train),y] = 0
  #print("margins  is:",margins)
  loss += np.sum(margins)/num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins >0] = 1
  #print("margins  is:",margins)
  row_num = np.sum(margins,axis = 1)
  margins [np.arange(num_train),y] = -row_num
  #print("margins  is:",margins)
  dW += np.dot(X.T,margins)/num_train + reg *W
  #print("dW  is:",dW)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
