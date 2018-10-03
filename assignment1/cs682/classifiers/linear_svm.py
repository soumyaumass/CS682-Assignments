import numpy as np
from random import shuffle

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
  for i in range(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in range(num_classes):
          if j == y[i]:
              continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
              loss += margin
              dW[:,j] += X[i]    
              dW[:,y[i]] -= X[i]
  
    # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += reg * np.sum(W * W)  # Add regularization to the loss
  
  dW /= num_train
  dW += 2 * reg*W       # Add regularization to gradient

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  
  loss = 0.0
  dW = np.zeros_like(W) # initialize the gradient as zero
  num_train = X.shape[0]

  score_matrix = np.matmul(X,W)   #calculating score matrix

  #calculating correct score matrix to do broadcast sum
  correct_score_vector = score_matrix[range(X.shape[0]),y]
  
  score_matrix += 1  # adding 1 to every element
  #calculating difference by broadcasting
  loss_matrix = score_matrix - correct_score_vector.reshape(X.shape[0],1)
  #removing all the negative losses
  loss_matrix[np.where(loss_matrix < 0)] = 0 
  
  loss = loss_matrix.sum().sum() - X.shape[0]
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  
  loss_matrix[np.where(loss_matrix == 1)] = 0
  loss_matrix[np.where(loss_matrix == 0.9999999999999999)] = 0
  
  margin_count_matrix = np.zeros_like(loss_matrix)
  margin_count_matrix[np.where(loss_matrix > 0)] = 1
  
  positive_margin_per_image =  margin_count_matrix.sum(axis=1,dtype=np.int32)
  
  #adding X[i] gradients
  dW = (np.matmul(margin_count_matrix.T,X)).T
  # Matrix containing summation of all the -1*X terms of dW
  margin_times_X = positive_margin_per_image.reshape(X.shape[0],1)*X
  
  #adding -X[i] for the correct classes
  for i in range(X.shape[0]):
      dW[:,y[i]] -= margin_times_X[i]

  dW /=  num_train
  dW += 2 * reg*W       # Add regularization to gradient    
 
  return loss, dW
