import pandas as pd
import dataSelection
import featureSelection
import preprocess
import serialization
import time
import os.path as path
import random
import torch

# TODO: add type annotations

# TODO: when using a binary model, check that health status count = 2
# TODO: read this from the command line
EXPERIMENT_DESCRIPTION_FILE = "experiments/test.toml"

experiment_config: serialization.ExperimentConfig = serialization.load_experiment(
    EXPERIMENT_DESCRIPTION_FILE
)

random.seed(experiment_config.seed)
torch.manual_seed(experiment_config.seed)

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
    experiment_config.good_bad_ratio,
    False,
)

print("Creating the AI model")
experiment_config.model.settings.evaluate_interval = 100

try:
    # TODO: pass the threshold here
    experiment_config.model.train_model(
        experiment_config.epoch_count,
        X_train,
        y_train,
        good_test,
        bad_test,
        experiment_config.loss_fn,
        experiment_config.optimizer,
        experiment_config.vote_count,
    )
except KeyboardInterrupt:
    pass
finally:
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S.txt")
    with open(path.join("results", timestr), "w") as f:
        f.write(experiment_config.to_string())
        f.write("\n#\n")
        f.write(",".join(map(str, experiment_config.model.loss)))
        f.write("\n#\n")
        # FAR
        f.write(",".join([str(x[0]) for x in experiment_config.model.failure_result]))
        f.write("\n#\n")
        # FDR
        f.write(",".join([str(x[1]) for x in experiment_config.model.failure_result]))
        f.write("\n#\n")
