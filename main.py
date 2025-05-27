import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
import dataSelection
import featureSelection
import preprocess
import bpnn
from utils import NNDescription
import serialization

# TODO: add type annotations

# TODO: when using a binary model, check that health status count = 2
# TODO: read this from the command line
EXPERIMENT_DESCRIPTION_FILE = "experiments/test.toml"

experiment_config: serialization.ExperimentConfig = serialization.load_experiment(
    EXPERIMENT_DESCRIPTION_FILE
)

print("Reading data file")

data: pd.DataFrame = pd.read_csv(experiment_config.data_file)

print("Computing change rates")
data = preprocess.computeChangeRates(data, experiment_config.change_rate_interval)

good_hard_drives: pd.DataFrame = data[data["Drive Status"] == 1]
bad_hard_drives: pd.DataFrame = preprocess.getLastSamples(
    data[data["Drive Status"] == -1], experiment_config.number_of_failing_samples
)

data = pd.concat([bad_hard_drives, good_hard_drives])

# TODO: check if the features change a lot when CHANGE_RATE_INTERVAL and NUMBER_OF_SAMPLES change
print(
    f"Selecting {experiment_config.feature_count} features using the {experiment_config.feature_selection_algorithm.name} algorithm"
)
data = featureSelection.selectFeatures(
    data,
    featureSelection.FeatureSelectionAlgorithm.Z_SCORE,
    experiment_config.feature_count,
)
print(f"Features kept: {str(list(data.columns)[2:])}")

print(
    f"Adding Health Status Values using {experiment_config.health_status_algorithm.name} algorithm"
)
bad_hard_drives = preprocess.addHealthStatus(
    bad_hard_drives,
    False,
    experiment_config.health_status_algorithm,
    experiment_config.health_status_count - 1,
)
good_hard_drives = preprocess.addHealthStatus(
    good_hard_drives,
    True,
    experiment_config.health_status_algorithm,
    experiment_config.health_status_count - 1,
)

print("Creating testing and training datasets")
X_train, y_train, good_test, bad_test = dataSelection.train_test(
    good_hard_drives,
    bad_hard_drives,
    experiment_config.seed,
    experiment_config.good_bad_ratio,
    False,
)

print("Creating the AI model")
# model = bpnn.BinaryBPNN(FEATURE_COUNT, HIDDEN_NODES)
model: bpnn.FailureDetectionNN = bpnn.MultiLevelBPNN(
    experiment_config.feature_count,
    experiment_config.hidden_nodes,
    experiment_config.health_status_count,
)
# model = bpnn.BinaryRNN(FEATURE_COUNT, HIDDEN_NODES)
# model = bpnn.MultiLevelRNN(FEATURE_COUNT, HIDDEN_NODES, HEALTH_STATUS_COUNT)
# model = bpnn.BinaryLSTM(FEATURE_COUNT, HIDDEN_NODES)
# model = bpnn.MultiLevelLSTM(FEATURE_COUNT, HIDDEN_NODES, HEALTH_STATUS_COUNT)
# loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss(weight=torch.tensor([1.0/(1.0+GOOD_BAD_RATIO)]))
# loss_fn = nn.BCELoss()
loss_fn: nn.Module = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer: torch.optim.Optimizer = optim.SGD(
    model.parameters(), lr=experiment_config.learning_rate, weight_decay=1e-7
)

if (
    model.description & NNDescription.BINARY
) and experiment_config.health_status_count != 2:
    raise ValueError(
        "When training a binary model, the number of classes must be equal to 2"
    )

if (
    model.description & NNDescription.MULTILEVEL
) and experiment_config.health_status_count < 3:
    raise ValueError(
        "When training a multi level model, the number of classes must be at least 3"
    )

model.validateDescription()
model.settings.evaluate_interval = 100

# TODO: pass the threshold here
model.train_model(
    experiment_config.epoch_count,
    X_train,
    y_train,
    good_test,
    bad_test,
    loss_fn,
    optimizer,
    experiment_config.vote_count,
    experiment_config.seed,
)
model.evaluate(
    good_test,
    bad_test,
    experiment_config.vote_count,
    experiment_config.seed,
    experiment_config.vote_threshold,
)
# bpnn.evaluate(model, complete_good, complete_bad, VOTE_COUNT)
