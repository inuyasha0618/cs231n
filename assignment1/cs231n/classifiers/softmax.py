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
  for i in xrange(X.shape[0]):
    maxF = np.dot(X[0], W[:, 0])
    f = np.zeros(W.shape[1])

    for j in xrange(W.shape[1]):
      f[j] = np.dot(X[i], W[:, j])
      if (f[j] > maxF):
        maxF = f[j]

    f -= maxF
    sumExp = np.sum(np.exp(f))
    loss += -f[y[i]] + np.log(sumExp)
    dW[:, y[i]] += -X[i]
    dW += 1 / sumExp * np.dot(X[i].reshape(X[i].shape[0], 1), np.exp(f).reshape(1, f.shape[0]))

  loss = loss / X.shape[0] + reg * np.sum(W * W)
  dW = dW / X.shape[0] + reg * 2 * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  f = np.dot(X, W)
  # expF = np.exp(f - f[np.arange(N), y][:, np.newaxis])
  expF = np.exp(f - f[np.arange(N), np.argmax(f, axis=1)][:, np.newaxis])
  sumExp = np.sum(expF, axis=1)[:, np.newaxis]
  probs = expF / sumExp
  probsArr = probs[np.arange(N), y]
  loss = np.sum(-np.log(probsArr)) / N + reg * np.sum(W * W)

  temp = np.zeros_like(f)
  temp[np.arange(N), y] = 1
  dW = -np.dot(X.T, temp) / N + np.dot((X / sumExp).T, expF) / N + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

