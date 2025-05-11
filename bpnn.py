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
    def train_model(self, x, y, epochs, loss_fn,optimizer, seed = 0):
      torch.manual_seed(seed)
      for epoch in range(epochs):
          self.train()
          optimizer.zero_grad()
          outputs = self(x).squeeze()
          loss = loss_fn(outputs, y.float())
          loss.backward()
          optimizer.step()
          if (epoch + 1) % 10 == 0:
              print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate_group(self, data, voteCount, ratio, target):
       with torch.no_grad():
        serialNumbers = data["serial-number"].unique()
        count = len(serialNumbers)
        y=data["Health Status"]
        X=data.drop(columns=["Health Status"])
        X = X.drop(columns=["Drive Status", "serial-number"])
        correct = 0
        for serialNumber in serialNumbers:
          indices = list(data[data["serial-number"] == serialNumber].index[-voteCount:])
          X_test= Variable(torch.from_numpy(np.array(X.loc[indices])).type(torch.FloatTensor))
          y_test= Variable(torch.from_numpy(np.array(y.loc[indices])).type(torch.LongTensor))
          
          if self.vote(X_test, ratio) == target:
            correct += 1
        
        return correct / count

    def vote(self, X_values, ratio = 0.5):
      """
      X_values correspond to a sequence of consecutive samples to a given hard drive
      The function returns 0 (the HD is considered as failing) if more than ratio of the samples are considered as failing, else it returns 1
      """
      predictions = self(X_values)
      predicted_classes = (predictions > 0.5).float().sum()
      return 1 if predicted_classes >= len(X_values) * (1-ratio) else 0

    def evaluate(self, data_good, data_bad, voteCount, seed = 0, ratio = 0.5):
      print(f"Evaluating model. A HD is considererd as failing if more than {ratio:.2f} of its samples are classified as failing.")
      torch.manual_seed(seed)
      self.eval()

      far = 1 - self.evaluate_group(data_good, voteCount, ratio, 1)
      fdr = self.evaluate_group(data_bad, voteCount, ratio, 0)

      print(f'FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%')