import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class BinaryClassifier(nn.Module):
    def __init__(self, input_count, hidden_nodes):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_count, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# TODO: add train and evaluate as member functions of the model
def train(x, y, model, epochs, loss_fn,optimizer):
  for epoch in range(epochs):
      model.train()
      optimizer.zero_grad()
      outputs = model(x).squeeze()
      loss = loss_fn(outputs, y.float())
      loss.backward()
      optimizer.step()
      if (epoch + 1) % 10 == 0:
          print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate(model, data_good, data_bad, voteCount):
  model.eval()
  #TODO: put this in a function
  with torch.no_grad():
    serialNumbers = data_good["serial-number"].unique()
    good_count = len(serialNumbers)
    y=data_good.pop("Health Status")
    X=data_good
    X = X.drop(columns=["Drive Status", "serial-number"])
    far = 0
    for serialNumber in serialNumbers:
      indices = list(data_good[data_good["serial-number"] == serialNumber].index[-voteCount:])
      X_test= Variable(torch.from_numpy(np.array(X.loc[indices])).type(torch.FloatTensor))
      y_test= Variable(torch.from_numpy(np.array(y.loc[indices])).type(torch.LongTensor))
      predictions = model(X_test)
      # TODO: extract this into a function
      predicted_classes = (predictions > 0.5).float().sum()
      if predicted_classes < voteCount / 2:
        far += 1
      # break
    
    far /= good_count

    serialNumbers = data_bad["serial-number"].unique()
    bad_count = len(serialNumbers)
    y=data_bad.pop("Health Status")
    X=data_bad
    X = X.drop(columns=["Drive Status", "serial-number"])
    fdr = 0
    for serialNumber in serialNumbers:
      indices = list(data_bad[data_bad["serial-number"] == serialNumber].index[-voteCount:])
      X_test= Variable(torch.from_numpy(np.array(X.loc[indices])).type(torch.FloatTensor))
      y_test= Variable(torch.from_numpy(np.array(y.loc[indices])).type(torch.LongTensor))
      predictions = model(X_test)
      # TODO: extract this into a function
      predicted_classes = (predictions > 0.5).float().sum()
      if predicted_classes < voteCount / 2:
        fdr += 1
    
    fdr /= bad_count
    print(f'FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%')