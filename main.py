import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataSelection
import featureSelection
import preprocess
import bpnn

#TODO: improve configuration
NUMBER_OF_SAMPLES = 24
DATA_FILE = "data/baidu-dataset.csv"
CHANGE_RATE_INTERVAL = 6
FEATURE_COUNT = 24
HEALTH_STATUS_COUNT = 6
VOTE_COUNT = 7
SEED = 0
EPOCH_COUNT = 400
LEARNING_RATE = 0.1
FEATURE_SELECTION_ALGORITHM = featureSelection.FeatureSelectionAlgorithm.Z_SCORE
HEALTH_STATUS_ALGORITHM = preprocess.HealthStatusAlgorithm.LINEAR
GOOD_BAD_RATIO = 5
HIDDEN_NODES = 32
VOTE_THRESHOLD = 0.5

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
X_train, y_train,  good_test, bad_test = dataSelection.train_test(good_hard_drives, bad_hard_drives, SEED, GOOD_BAD_RATIO, False)

print("Creating the AI model")
# model = bpnn.BinaryClassifier(FEATURE_COUNT, HIDDEN_NODES)
# model = bpnn.MultiLevelClassifier(FEATURE_COUNT, HIDDEN_NODES, HEALTH_STATUS_COUNT)
# model = bpnn.BinaryRNN(FEATURE_COUNT, HIDDEN_NODES)
# model = bpnn.MultiLevelRNN(FEATURE_COUNT, HIDDEN_NODES, HEALTH_STATUS_COUNT)
model = bpnn.MultiLevelLSTM(FEATURE_COUNT, HIDDEN_NODES, HEALTH_STATUS_COUNT)
# model = bpnn.BinaryLSTM(FEATURE_COUNT, HIDDEN_NODES)
# loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss(weight=torch.tensor([1.0/(1.0+GOOD_BAD_RATIO)]))
# loss_fn = nn.BCELoss()
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)


model.train_model(X_train, y_train, EPOCH_COUNT, loss_fn, optimizer, good_test, bad_test, VOTE_COUNT, SEED)   
model.evaluate(good_test, bad_test, VOTE_COUNT, SEED, VOTE_THRESHOLD)
# bpnn.evaluate(model, complete_good, complete_bad, VOTE_COUNT)