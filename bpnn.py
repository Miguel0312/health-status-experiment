import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from enum import Enum
import random

class OutputType(Enum):
  INTEGER = 1
  FLOAT = 2

class FailureDetectionNN(nn.Module):
  def __init__(self):
    super(FailureDetectionNN, self).__init__()
    self.output_type = OutputType.INTEGER

  def forward(self, x):
    return self.net(x)

  def train_model(self, x, y, epochs, loss_fn, optimizer, seed=0):
    torch.manual_seed(seed)
    # print(x)
    # print(y)
    # x = x.drop(["serial-number", "Drive Status"] , axis = 1)
    # x = Variable(torch.from_numpy(np.array(x)).type(torch.FloatTensor))
    # y = Variable(torch.from_numpy(np.array(y)).type(torch.LongTensor))

    for epoch in range(epochs):
      self.train()
      optimizer.zero_grad()
      outputs = self(x).squeeze()
      if self.output_type == OutputType.FLOAT:
        loss = loss_fn(outputs, y.float())
      elif self.output_type == OutputType.INTEGER:
        loss = loss_fn(outputs, y)
      loss.backward()
      optimizer.step()
      if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

  def evaluate_group(self, data, voteCount, ratio, target):
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

        if self.vote(X_test, ratio) == target:
            correct += 1

    return correct / count

  def evaluate(self, data_good, data_bad, voteCount, seed=0, ratio=0.5):
    print(
        f"Evaluating model. A HD is considererd as failing if more than {ratio:.2f} of its samples are classified as failing."
    )
    torch.manual_seed(seed)
    self.eval()

    far = 1 - self.evaluate_group(data_good, voteCount, ratio, 1)
    fdr = self.evaluate_group(data_bad, voteCount, ratio, 0)

    print(f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%")


class BinaryClassifier(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes):
    super(BinaryClassifier, self).__init__()
    self.output_type = OutputType.FLOAT
    self.net = nn.Sequential(
      nn.Linear(input_count, hidden_nodes),
      nn.ReLU(),
      nn.Linear(hidden_nodes, 1),
      nn.Sigmoid(),
    )

  def vote(self, X_values, ratio=0.5):
    """
    X_values correspond to a sequence of consecutive samples to a given hard drive
    The function returns 0 (the HD is considered as failing) if more than ratio of the samples are considered as failing, else it returns 1
    """
    predictions = self(X_values)
    predicted_classes = (predictions > 0.5).float().sum()
    return 1 if predicted_classes >= len(X_values) * (1 - ratio) else 0

class MultiLevelClassifier(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes, output_count):
    super(MultiLevelClassifier, self).__init__()
    self.output_type = OutputType.INTEGER
    self.net = nn.Sequential(
      nn.Linear(input_count, hidden_nodes),
      nn.ReLU(),
      nn.Linear(hidden_nodes, output_count),
      nn.Sigmoid(),
    )
    self.first = True

  def vote(self, X_values, ratio=0.5):
    """
    X_values correspond to a sequence of consecutive samples to a given hard drive
    The function returns 0 (the HD is considered as failing) if more than ratio of the samples are considered as failing, else it returns 1
    """
    predictions = self(X_values)
    if self.first:
      self.first = False
    # Algorithm described by Health Status Assessment and Failure Prediction for Hard Drives with Recurrent Neural Networks
    predictions = predictions.sum(dim=0)
    bad = predictions[0:-2].sum()
    good = predictions[-1].item()
    return 1 if good >= bad else 0
  
class BinaryRNN(FailureDetectionNN):
  def __init__(self, input_count, hidden_nodes):
    super(BinaryRNN, self).__init__()

    self.hidden_nodes = hidden_nodes

    self.net = nn.RNN(input_count, hidden_nodes,nonlinearity='relu')
    self.h2o = nn.Linear(hidden_nodes, 1)
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.LogSoftmax(dim=0)

  def forward(self, x):
    rnn_out, hidden = self.net(x)
    output = self.h2o(hidden[0])
    output = self.sigmoid(output)
    return output

  def train_model(self, x, y, epochs, loss_fn, optimizer, data_good, data_bad, voteCount, seed=0):
    torch.manual_seed(seed)
    self.train()

    serialNumbers = x["serial-number"].unique()
    y = Variable(torch.from_numpy(np.array(y)).type(torch.FloatTensor))

    batches = []
    answer = []
    index = 0

    for serialNumber in serialNumbers:
      answer.append(y[index])
      hd_data = x[x["serial-number"] == serialNumber]
      index += len(hd_data)
      hd_data = hd_data.drop(["serial-number"], axis = 1).tail(12)
      batches.append(Variable(torch.from_numpy(np.array(hd_data)).type(torch.FloatTensor)))

    permutation = list(range(len(batches)))
    random.shuffle(permutation)
    batches = [batches[permutation[i]] for i in range(len(batches))]
    answer = [answer[permutation[i]] for i in range(len(answer))]

    for epoch in range(epochs):
      self.zero_grad()
      self.net.zero_grad()

      current_loss = 0
      for idx, batch in enumerate(batches):
        output = self(batch).squeeze()

        loss = loss_fn(output, answer[idx])
        current_loss += loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()

      current_loss /= len(batches)
          
      if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
      if (epoch + 1) % 10 == 0:
        self.evaluate(data_good, data_bad, voteCount)
      if(epoch+1) %100 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2

  def evaluate_group(self, data, voteCount, target):
    y = data["Health Status"]
    x = data.drop(columns=["Health Status", "Drive Status"], axis=1)

    serialNumbers = data["serial-number"].unique()
    y = Variable(torch.from_numpy(np.array(y)).type(torch.LongTensor))
    batches = []
    for serialNumber in serialNumbers:
      hd_data = x[x["serial-number"] == serialNumber]
      hd_data = hd_data.drop(["serial-number"], axis = 1).tail(voteCount)
      batches.append(Variable(torch.from_numpy(np.array(hd_data)).type(torch.FloatTensor)))

    correct = 0
    for idx, batch in enumerate(batches):
      output = self(batch)
      if round(output.item()) == target:
        correct += 1
    
    correct /= len(batches)
    return correct

  def evaluate(self, data_good, data_bad, voteCount, seed=0, ratio=0.5):
    torch.manual_seed(seed)
    self.eval()
    
    far = 1 - self.evaluate_group(data_good, voteCount, 1)
    fdr = self.evaluate_group(data_bad, voteCount, 0)

    print(f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%")