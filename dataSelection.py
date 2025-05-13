import random
import pandas as pd
import torch
from torch.autograd import Variable
import numpy as np

def train_test(good_hard_drives, bad_hard_drives, seed = 0, good_bad_ratio = 1, ratioOverSerialNumbers = False):
  random.seed(seed)
  ratio = bad_hard_drives.size / good_hard_drives.size

  serial_number_bad = list(bad_hard_drives["serial-number"].unique())
  serial_number_good = list(good_hard_drives["serial-number"].unique())

  if ratioOverSerialNumbers:
    ratio = len(serial_number_bad) / len(serial_number_good)

  serial_number_bad_sample = random.sample(serial_number_bad, int(0.8 * len(serial_number_bad)))
  bad_train = bad_hard_drives[bad_hard_drives["serial-number"].isin(serial_number_bad_sample)]
  bad_test = bad_hard_drives[~bad_hard_drives["serial-number"].isin(serial_number_bad_sample)]

  serial_number_good_train = random.sample(serial_number_good, int(0.8 * ratio * good_bad_ratio * len(serial_number_good)))
  serial_number_good_test = list(set(serial_number_good) - set(serial_number_good_train))
  good_train = good_hard_drives[good_hard_drives["serial-number"].isin(serial_number_good_train)]
  good_test = good_hard_drives[good_hard_drives["serial-number"].isin(serial_number_good_test)]

  print(len(serial_number_good_train), len(serial_number_bad_sample))

  df = pd.concat([good_train, bad_train])
  df = df.drop(["Drive Status"], axis = 1)

  y=df.pop("Health Status")
  X=df

  print(len(X))

  # Convert data to PyTorch tensors
  # X_train= Variable(torch.from_numpy(np.array(X)).type(torch.FloatTensor))
  # y_train= Variable(torch.from_numpy(np.array(y)).type(torch.LongTensor))

  return X, y, good_test, bad_test
  # return X_train, y_train, good_test, bad_test
