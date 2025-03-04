import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
     scores = X[i].dot(W)
     shift_scores = scores - np.max(scores)
     scores_expsum = np.sum(np.exp(shift_scores))
     score_exp = np.exp(shift_scores[y[i]])
     loss += -np.log(score_exp/scores_expsum)

     #loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
     #loss += loss_i
     for j in xrange(num_classes):
         softmax_output = np.exp(shift_scores[j])/scores_expsum
         if j == y[i]:
             dW[:,j] += (-1 + softmax_output) *X[i] 
         else: 
             dW[:,j] += softmax_output *X[i] 
  
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  dW = dW/num_train + reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1,keepdims =True )
  scores_expsum = np.sum(np.exp(shift_scores), axis = 1 ,keepdims =True)

  #score_exp = np.exp(shift_scores[y[i]])
  softmax_output = np.exp(shift_scores)/scores_expsum

  loss = -np.sum(np.log(softmax_output[np.arange(num_train), y]))

  ind = np.zeros_like(softmax_output)
  ind[np.arange(num_train),y] = 1
  #print("ind is ",ind)
  dW = X.T.dot(softmax_output-ind)

  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  dW = dW/num_train + reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

