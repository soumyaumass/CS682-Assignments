import numpy as np
from random import shuffle

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
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
      scores = X[i].dot(W)
      denom = 0
      numer = scores[y[i]]
      denom = np.sum(np.exp(scores))
      for j in range(num_classes):
          if(j != y[i]):
              dW[:,j] += (np.exp(scores[j]) * X[i])/denom
          else:
              dW[:,j] += X[i] * (np.exp(scores[j])/denom - 1)
          
      loss -= numer - np.log(denom)
              
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += reg * np.sum(W * W)  # Add regularization to the loss
  
  dW /= num_train
  dW += 2 * reg*W       # Add regularization to gradient

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  score_matrix = np.exp(np.matmul(X,W))   #calculating score matrix
  
  log_score_sum_denom = np.log(score_matrix.sum(axis=1)).sum()
  
  #calculating correct score matrix to do broadcast sum
  correct_score_vector = score_matrix[range(X.shape[0]),y]
 
  log_score_sum_numer = (-1) * np.log(correct_score_vector).sum()
  log_score_sum = log_score_sum_denom + log_score_sum_numer
  
  loss = log_score_sum / num_train
  loss += reg * np.sum(W * W) # Add regularization to the loss
  
  main_matrix = score_matrix/(score_matrix.sum(axis=1)).reshape(score_matrix.shape[0],1)
  main_matrix[range(main_matrix.shape[0]),y] -= 1
  dW = np.matmul(X.T,main_matrix)
  
  dW /= num_train
  dW += 2 * reg*W   # Add regularization to the gradient
  
  return loss, dW

  


