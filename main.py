import pandas as pd
import dataSelection
import featureSelection
import modelBase
import preprocess
import serialization
import time
import os.path as path
import random
import torch
import sys
import numpy as np

from utils import NNDescription

if len(sys.argv) < 2:
    print("Usage: python3 main.py experiment1 [experiment2 experiment3 ...]")
    exit(0)

# TODO: read this from the command line arguments or from the config file
vote_test = False
compute_change_rates = True

# TODO: deduce this from the config files
attributes = [
    "feature_count",
    "vote_threshold",
    "feature_count",
    "good_bad_ratio",
    "max_depth",
    "min_samples_leaf",
]

for fileIDX, file_name in enumerate(sys.argv[1:]):
    experiment_config: serialization.ExperimentConfig = serialization.load_experiment(
        file_name
    )

    results = []

    for i in range(len(experiment_config.model)):
        random.seed(experiment_config.seed[i])
        torch.manual_seed(experiment_config.seed[i])
        np.random.seed(experiment_config.seed[i])

        # print("Reading data file")

        data: pd.DataFrame = pd.read_csv(experiment_config.data_file[i])

        if compute_change_rates:
            # print("Computing change rates")
            data = preprocess.computeChangeRates(
                data, experiment_config.change_rate_interval[i]
            )

        # TODO: check if the features change a lot when CHANGE_RATE_INTERVAL and NUMBER_OF_SAMPLES change
        data = featureSelection.selectFeatures(
            data,
            featureSelection.FeatureSelectionAlgorithm.Z_SCORE,
            experiment_config.feature_count[i],
        )
        assert len(list(data.columns)[2:]) == experiment_config.feature_count[i]

        good_hard_drives: pd.DataFrame = data[data["Drive Status"] == 1]
        bad_hard_drives: pd.DataFrame = data[data["Drive Status"] == -1]

        bad_hard_drives = preprocess.addHealthStatus(
            bad_hard_drives,
            False,
            experiment_config.health_status_algorithm[i],
            experiment_config.health_status_count[i] - 1,
        )
        good_hard_drives = preprocess.addHealthStatus(
            good_hard_drives,
            True,
            experiment_config.health_status_algorithm[i],
            experiment_config.health_status_count[i] - 1,
        )

        # print("Creating testing and training datasets")
        X_train, y_train, good_test, bad_test = dataSelection.train_test(
            good_hard_drives,
            bad_hard_drives,
            experiment_config.good_bad_ratio[i],
            (
                experiment_config.model_type == modelBase.ModelType.NN
                and experiment_config.model[i].description & NNDescription.TEMPORAL
            )
            != 0,
            experiment_config.number_of_failing_samples[i],
        )

        # X_train = preprocess.getLastSamples(
        #     X_train,
        #     experiment_config.number_of_failing_samples[i],
        # )

        # print("Training the AI model")

        try:
            match experiment_config.model_type:
                case modelBase.ModelType.NN:
                    experiment_config.model[i].train_model(
                        experiment_config.epoch_count[i],
                        X_train,
                        y_train,
                        good_test,
                        bad_test,
                        experiment_config.loss_fn[i],
                        experiment_config.optimizer[i],
                        experiment_config.vote_count[i],
                    )
                case modelBase.ModelType.TREE:
                    experiment_config.model[i].train_model(
                        X_train,
                        y_train,
                        good_test,
                        bad_test,
                        experiment_config.vote_count[i],
                    )
                case _:
                    raise ValueError("Invalid Value of model_type")
        except KeyboardInterrupt:
            pass
        finally:
            results.append(experiment_config.model[i].failure_result[-1])

            timestr = time.strftime("%Y_%m_%d-%H_%M_%S.txt")
            with open(path.join("results", timestr), "w") as f:
                f.write(experiment_config.print_experiment(i))
                f.write("\n#\n")
                if experiment_config.model_type == modelBase.ModelType.NN:
                    f.write(",".join(map(str, experiment_config.model[i].loss)))
                f.write("\n#\n")
                # FAR
                f.write(
                    ",".join(
                        [str(x[0]) for x in experiment_config.model[i].failure_result]
                    )
                )
                f.write("\n#\n")
                # TIA
                f.write(
                    ",".join(
                        [str(x[1]) for x in experiment_config.model[i].failure_result]
                    )
                )
                f.write("\n#\n")
                f.write(
                    ",".join(
                        [str(x[2]) for x in experiment_config.model[i].failure_result]
                    )
                )
                f.write("\n#\n")
                f.write(
                    ",".join(
                        [str(x[3]) for x in experiment_config.model[i].failure_result]
                    )
                )
                f.write("\n#\n")

            # If it is a vote_test, we train only one model
            if vote_test:
                break

    if vote_test:
        experiment_config.model[0].failure_result = []
        # Evaluate the same model with the other voting parameters
        for i in range(1, len(experiment_config.model)):
            # Always use the same model
            experiment_config.model[0].evaluate(
                good_test,
                bad_test,
                experiment_config.vote_count[i],
                experiment_config.vote_threshold[i],
            )
            results.append(experiment_config.model[0].failure_result[0])
            experiment_config.model[0].failure_result = []

    print("\n-------------Results----------------")
    for result in results:
        print(
            f"FAR: {100*result[0]:.3f}%, FDR: {100*result[1]:.3f}%, TIA: {result[2]:.3f}, TIA Std Dev: {result[3]:.3f}"
        )

    attribute = attributes[fileIDX]

    print(f"|{attribute}|FAR(%)|FDR(%)|TIA(h)|TIA SD(h)|")
    print("|-------------|------|------|------|---------|")
    for idx, result in enumerate(results):
        try:
            val = getattr(experiment_config, attribute)[idx]
        except Exception:
            # If the attribute was not found on the config, search on the model object directly
            val = getattr(experiment_config.model[idx].settings, attribute)
        print(
            f"|{val}|{100*result[0]:.2f}|{100*result[1]:.2f}|{result[2]:.1f}|{result[3]:.1f}|"
        )
