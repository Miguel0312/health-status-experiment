import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import featureSelection
import preprocess
import bpnn
from sklearn.model_selection import train_test_split
import random

#TODO: improve configuration
NUMBER_OF_SAMPLES = 24
DATA_FILE = "data/baidu-dataset.csv"
CHANGE_RATE_INTERVAL = 6
FEATURE_COUNT = 24
HEALTH_STATUS_COUNT = 2
VOTE_COUNT = 18
SEED = 0
EPOCH_COUNT = 400
LEARNING_RATE = 0.01
FEATURE_SELECTION_ALGORITHM = featureSelection.FeatureSelectionAlgorithm.Z_SCORE
HEALTH_STATUS_ALGORITHM = preprocess.HealthStatusAlgorithm.LINEAR
GOOD_BAD_RATIO = 1
HIDDEN_NODES = 64

torch.manual_seed(SEED)

print("Reading data file")

data = pd.read_csv(DATA_FILE)

print("Computing change rates")
data = preprocess.computeChangeRates(data, CHANGE_RATE_INTERVAL)

good_hard_drives = data[data["Drive Status"] == 1]
bad_hard_drives = preprocess.getLastSamples(data[data["Drive Status"] == -1], NUMBER_OF_SAMPLES)

data = pd.concat([bad_hard_drives, good_hard_drives])

# TODO: check if the features change a lot when CHANGE_RATE_INTERVAL and NUMBER_OF_SAMPLES change
print(f"Selecting {FEATURE_COUNT} features using the {FEATURE_SELECTION_ALGORITHM.name} algorithm")
data = featureSelection.selectFeatures(data, featureSelection.FeatureSelectionAlgorithm.Z_SCORE, FEATURE_COUNT)
print(f"Features kept: {str(list(data.columns)[2:])}" )

good_hard_drives = data[data["Drive Status"] == 1]
bad_hard_drives = preprocess.getLastSamples(data[data["Drive Status"] == -1], NUMBER_OF_SAMPLES)

print(f"Adding Health Status Values using {HEALTH_STATUS_ALGORITHM.name} algorithm")
bad_hard_drives = preprocess.addHealthStatus(bad_hard_drives, False, HEALTH_STATUS_ALGORITHM, HEALTH_STATUS_COUNT-1)
good_hard_drives = preprocess.addHealthStatus(good_hard_drives, True, HEALTH_STATUS_ALGORITHM, HEALTH_STATUS_COUNT-1)

print("Creating testing and training datasets")
complete_good = good_hard_drives
complete_bad = bad_hard_drives
print(complete_bad.columns)

ratio = bad_hard_drives.size / good_hard_drives.size

good_hard_drives_sample = good_hard_drives.sample(frac=GOOD_BAD_RATIO*ratio)

serial_number_bad = list(bad_hard_drives["serial-number"].unique())
serial_number_bad_sample = random.sample(serial_number_bad, int(0.8 * len(serial_number_bad)))
bad_train = bad_hard_drives[bad_hard_drives["serial-number"].isin(serial_number_bad_sample)]
bad_test = bad_hard_drives[~bad_hard_drives["serial-number"].isin(serial_number_bad_sample)]

serial_number_good = list(good_hard_drives_sample["serial-number"].unique())
serial_number_good_sample = random.sample(serial_number_good, int(0.8 * len(serial_number_good)))
good_train = good_hard_drives_sample[good_hard_drives_sample["serial-number"].isin(serial_number_good_sample)]
good_test = good_hard_drives[~good_hard_drives["serial-number"].isin(serial_number_good_sample)]

df = pd.concat([good_train, bad_train])

df = df.drop(["serial-number", "Drive Status"] , axis = 1)

y=df.pop("Health Status")
X=df

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# good_test = good_hard_drives.loc[X_train.index]

# Convert data to PyTorch tensors
X_train= Variable(torch.from_numpy(np.array(X)).type(torch.FloatTensor))
# X_test= Variable(torch.from_numpy(np.array(y)).type(torch.FloatTensor))
y_train= Variable(torch.from_numpy(np.array(y)).type(torch.LongTensor))
# y_test= Variable(torch.from_numpy(np.array(y_test)).type(torch.LongTensor))

model = bpnn.BinaryClassifier(FEATURE_COUNT, HIDDEN_NODES)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

bpnn.train(X_train, y_train, model, EPOCH_COUNT, loss_fn, optimizer)
bpnn.evaluate(model, good_test, bad_test, VOTE_COUNT)
# bpnn.evaluate(model, complete_good, complete_bad, VOTE_COUNT)