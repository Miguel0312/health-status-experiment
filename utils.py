from enum import IntFlag, auto
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import random

class NNDescription(IntFlag):
  BINARY = auto()
  MULTILEVEL = auto()
  TEMPORAL = auto()
  UNIQUE = auto()
  LSTM = auto()
  RNN = auto()
  BP = auto()

def train(model, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0):
  torch.manual_seed(seed)
  model.train()
  if model.description & NNDescription.BP:
    _train_bp(model, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0)
  elif model.description & NNDescription.TEMPORAL:
    _train_temporal(model, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0)
  
def evaluate(model, data_good, data_bad, voteCount, seed=0, ratio=0.5):
  torch.manual_seed(seed)
  model.eval()

  far = 1 - _evaluate_group(model, data_good, voteCount, ratio, 1)
  fdr = _evaluate_group(model, data_bad, voteCount, ratio, 0)

  print(f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%")

def _evaluate_group(model, data, voteCount, ratio, target):
  with torch.no_grad():
    serialNumbers = data["serial-number"].unique()
    count = len(serialNumbers)
    y = data["Health Status"]
    X = data.drop(columns=["Health Status", "Drive Status", "serial-number"], axis=1)
    correct = 0
    for serialNumber in serialNumbers:
      indices = list(
          data[data["serial-number"] == serialNumber].index[-voteCount:]
      )
      X_test = Variable(
          torch.from_numpy(np.array(X.loc[indices])).type(torch.FloatTensor)
      )
      y_test = Variable(
          torch.from_numpy(np.array(y.loc[indices])).type(torch.LongTensor)
      )

      if _vote(model, X_test, ratio) == target:
        correct += 1

  return correct / count


def _vote(model, X_values, ratio):
  """
    X_values correspond to a sequence of consecutive samples to a given hard drive
    The function returns 0 (the HD is considered as failing) if more than ratio of the samples are considered as failing, else it returns 1
  """
  if model.description & NNDescription.BINARY:
    return _vote_binary(model, X_values, ratio)
  elif model.description & NNDescription.MULTILEVEL:
    return _vote_multilevel(model, X_values, ratio)
  
def _train_bp(model, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0):
    x = train_x.drop(["serial-number"] , axis = 1)
    x = Variable(torch.from_numpy(np.array(x)).type(torch.FloatTensor))

    model.settings.verify()

    if model.description & NNDescription.BINARY:
        y = Variable(torch.from_numpy(np.array(train_y)).type(torch.FloatTensor))
    elif model.description & NNDescription.MULTILEVEL:
      y = Variable(torch.from_numpy(np.array(train_y)).type(torch.LongTensor))

    for epoch in range(epochs):
      model.train()
      optimizer.zero_grad()
      outputs = model(x).squeeze()
      loss = loss_fn(outputs, y)
      loss.backward()
      optimizer.step()
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
      if(epoch+1) % model.settings.lr_decay_interval == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
      if (epoch + 1) % model.settings.evaluate_interval == 0:
        model.evaluate(test_good, test_bad, voteCount)

def _train_temporal(model, epochs, train_x, train_y, test_good, test_bad, loss_fn, optimizer, voteCount, seed=0):
  serialNumbers = train_x["serial-number"].unique()

  if model.description & NNDescription.BINARY:
    y = Variable(torch.from_numpy(np.array(train_y)).type(torch.FloatTensor))
  elif model.description & NNDescription.MULTILEVEL:
    y = Variable(torch.from_numpy(np.array(train_y)).type(torch.LongTensor))

  batches = []
  answer = []
  index = 0
  lookback = voteCount

  for serialNumber in serialNumbers:
    hd_data = train_x[train_x["serial-number"] == serialNumber]
    hd_data = hd_data.drop(["serial-number"], axis = 1)
    hd_data = Variable(torch.from_numpy(np.array(hd_data)).type(torch.FloatTensor))
    i = len(hd_data) - lookback - 1
    batches.append(hd_data[i:i+lookback])
    answer.append(y[index+i+1:index+lookback+i+1])

    index += len(hd_data)

  # Need to shuffle the batches because backpropagation is doen after each batch
  # There may be errors if the model is trained with a long sequence of samples with the same output
  permutation = list(range(len(batches)))
  random.shuffle(permutation)
  batches = [batches[permutation[i]] for i in range(len(batches))]
  answer = [answer[permutation[i]] for i in range(len(answer))]

  for epoch in range(epochs):
    model.zero_grad()
    model.net.zero_grad()

    current_loss = 0
    for idx, batch in enumerate(batches):
      output = model(batch).squeeze()

      loss = loss_fn(output, answer[idx])
      current_loss += loss

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 3)
      optimizer.step()
      optimizer.zero_grad()

    current_loss /= len(batches)
        
    if (epoch + 1) % 1 == 0:
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
    if (epoch + 1) % 10 == 0:
      model.evaluate(test_good, test_bad, voteCount)
    if(epoch+1) %100 == 0:
      optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2

def _vote_binary(model, X_values, ratio):
  predictions = model(X_values).squeeze()
  predicted_classes = (predictions > 0.5).float().sum().item()
  return 1 if predicted_classes >= len(X_values) * (1 - ratio) else 0

def _vote_multilevel(model, X_values, ratio):
  predictions = model(X_values)
  # Algorithm described by Health Status Assessment and Failure Prediction for Hard Drives with Recurrent Neural Networks
  good = 0
  for pred in predictions:
    pred = nn.Softmax(dim = 0)(pred)
    if pred[:-2].sum() < pred[-1]:
      good += 1
  return 1 if good >= len(X_values)*ratio else 0