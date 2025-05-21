import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from enum import Enum
from utils import NNDescription
import utils
import random

class ModelSettings():
  def __init__(self, input_count = None, hidden_nodes = None, output_count = None, evaluate_interval = 10, lr_decay_interval = 100):
    self.input_count = input_count
    self.hidden_nodes = hidden_nodes
    self.output_count = output_count
    self.evaluate_interval = evaluate_interval
    self.lr_decay_interval = lr_decay_interval

  def verify(self):
    if self.input_count is None:
      raise ValueError("Must specify number of inputs")
    if self.output_count is None:
      raise ValueError("Must specify number of outputs")
    if self.hidden_nodes is None:
      raise ValueError("Must specify number of hidden nodes")

class FailureDetectionNN(nn.Module):
  def __init__(self, input_count, hidden_nodes, output_count, evaluate_interval = 10, lr_decay_interval = 100):
    super(FailureDetectionNN, self).__init__()
    self.settings = ModelSettings(input_count, hidden_nodes, output_count, evaluate_interval, lr_decay_interval)
    self.description = 0

  def train_model(self, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0):
    utils.train(self,  epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed)

  def evaluate(self, data_good, data_bad, voteCount, seed=0, ratio=0.5):
    utils.evaluate(self, data_good, data_bad, voteCount, seed, ratio)

  def validateDescription(self):
    if self.description == 0:
      raise ValueError("No description entered for the model")
    isBinary = self.description & NNDescription.BINARY
    isMultiLevel = self.description & NNDescription.MULTILEVEL
    isTemporal = self.description & NNDescription.TEMPORAL
    isUnique = self.description & NNDescription.UNIQUE
    isLSTM = self.description & NNDescription.LSTM
    isRNN = self.description & NNDescription.RNN
    isBP = self.description & NNDescription.BP

    if isBinary and isMultiLevel:
      raise ValueError("The model must be either binary or multilevel, but not both")
    if (not isBinary) and (not isMultiLevel):
      raise ValueError("The model must be one of binary or multilevel")
    if isTemporal and isUnique:
      raise ValueError("The model must be either unique or temporal, but not both")
    if (not isTemporal) and (not isUnique):
      raise ValueError("The model must be one of unique or temporal")
    if isUnique and not isBP:
      raise ValueError("A unique model must be temporal")
    if isTemporal and (not isLSTM and not isRNN):
      raise ValueError("A temporal model must be either an LSTM or an RNN")
    if (isBP |isRNN |isLSTM).bit_count() != 1:
      raise ValueError("A description must be exactly one of BP, RNN and LSTM")


class BinaryBPNN(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes):
    super(BinaryBPNN, self).__init__(input_count, hidden_nodes, 1)
    self.description |= NNDescription.BP | NNDescription.BINARY | NNDescription.UNIQUE
    self.net = nn.Sequential(
      nn.Linear(input_count, hidden_nodes),
      nn.ReLU(),
      nn.Linear(hidden_nodes, 1),
      nn.Sigmoid(),
    )
  
  def forward(self, x):
    return self.net(x)

class MultiLevelBPNN(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes, output_count):
    super(MultiLevelBPNN, self).__init__(input_count, hidden_nodes, output_count)
    self.description |= NNDescription.BP | NNDescription.MULTILEVEL | NNDescription.UNIQUE
    self.output_count = output_count
    self.net = nn.Sequential(
      nn.Linear(input_count, hidden_nodes),
      nn.ReLU(),
      nn.Linear(hidden_nodes, output_count),
      # nn.Sigmoid(),
      # nn.Softmax(dim=1)
    )
  
  def forward(self, x):
    return self.net(x)

class BinaryRNN(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes):
    super(BinaryRNN, self).__init__(input_count, hidden_nodes, 2)

    self.description |= NNDescription.BINARY | NNDescription.TEMPORAL | NNDescription.RNN
    self.net = nn.RNN(input_count, hidden_nodes, nonlinearity='relu')
    self.linear = nn.Linear(hidden_nodes, 1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    output, _ = self.net(x)
    output = self.linear(output)
    output = self.sigmoid(output)
    return output

class MultiLevelRNN(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes, output_count):
    super(MultiLevelRNN, self).__init__(input_count, hidden_nodes, output_count)

    self.description |= NNDescription.MULTILEVEL | NNDescription.TEMPORAL | NNDescription.RNN
    self.net = nn.RNN(input_count, hidden_nodes, nonlinearity='relu')
    self.linear = nn.Linear(hidden_nodes, output_count)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    output, _ = self.net(x)
    output = self.linear(output)
    output = self.sigmoid(output)
    return output

class BinaryLSTM(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes):
    super(BinaryLSTM, self).__init__(input_count, hidden_nodes, 2)

    self.description |= NNDescription.BINARY | NNDescription.LSTM | NNDescription.TEMPORAL
    self.net = nn.LSTM(input_count, hidden_nodes, batch_first=True)
    self.linear = nn.Linear(hidden_nodes, 1)
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.LogSoftmax(dim=0)

  def forward(self, x):
    output, _ = self.net(x)
    output = self.linear(output)
    output = self.sigmoid(output)
    return output

class MultiLevelLSTM(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes, output_count):
    super(MultiLevelLSTM, self).__init__(input_count, hidden_nodes, output_count)

    self.description |= NNDescription.MULTILEVEL | NNDescription.LSTM | NNDescription.TEMPORAL
    self.net = nn.LSTM(input_count, hidden_nodes,batch_first=True)
    self.linear = nn.Linear(hidden_nodes, output_count)
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.LogSoftmax(dim=0)

  def forward(self, x):
    output, _ = self.net(x)
    output = self.linear(output)
    output = self.sigmoid(output)
    return output