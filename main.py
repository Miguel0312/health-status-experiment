import pandas as pd
import dataSelection
import featureSelection
import preprocess
import serialization
import time
import os.path as path
import random
import torch
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python3 main.py experiment1 [experiment2 experiment3 ...]")
    exit(0)

for file_name in sys.argv[1:]:
    experiment_config: serialization.ExperimentConfig = serialization.load_experiment(
        file_name
    )

    results = []
    # TODO: read this from the command line arguments
    vote_test = False

    # TODO: separate the data that can change from one experiment to the other from the one that can't
    for i in range(len(experiment_config.model)):
        random.seed(experiment_config.seed[i])
        torch.manual_seed(experiment_config.seed[i])
        np.random.seed(experiment_config.seed[i])

        print("Reading data file")

        data: pd.DataFrame = pd.read_csv(experiment_config.data_file[i])

        print("Computing change rates")
        # data = preprocess.computeChangeRates(
        #     data, experiment_config.change_rate_interval[i]
        # )

        # good_hard_drives: pd.DataFrame = data[data["Drive Status"] == 1]
        # bad_hard_drives = data[data["Drive Status"] == -1]
        # bad_hard_drives: pd.DataFrame = preprocess.getLastSamples(
        #     data[data["Drive Status"] == -1],
        #     experiment_config.number_of_failing_samples[i],
        # )

        # data = pd.concat([bad_hard_drives, good_hard_drives])

        # TODO: check if the features change a lot when CHANGE_RATE_INTERVAL and NUMBER_OF_SAMPLES change
        print(
            f"Selecting {experiment_config.feature_count[i]} features using the {experiment_config.feature_selection_algorithm[i].name} algorithm"
        )
        data = featureSelection.selectFeatures(
            data,
            featureSelection.FeatureSelectionAlgorithm.Z_SCORE,
            experiment_config.feature_count[i],
        )
        print(f"Features kept: {str(list(data.columns)[2:])}")

        print(
            f"Adding Health Status Values using {experiment_config.health_status_algorithm[i].name} algorithm"
        )

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

        print("Creating testing and training datasets")
        X_train, y_train, good_test, bad_test = dataSelection.train_test(
            good_hard_drives,
            bad_hard_drives,
            experiment_config.good_bad_ratio[i],
            experiment_config.number_of_failing_samples[i],
        )

        # X_train = preprocess.getLastSamples(
        #     X_train,
        #     experiment_config.number_of_failing_samples[i],
        # )

        print("Training the AI model")

        try:
            # TODO: pass the threshold here
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
        except KeyboardInterrupt:
            pass
        finally:
            if not vote_test:
                results.append(experiment_config.model[i].failure_result[-1])
            else:
                break

            timestr = time.strftime("%Y_%m_%d-%H_%M_%S.txt")
            with open(path.join("results", timestr), "w") as f:
                f.write(experiment_config.print_experiment(i))
                f.write("\n#\n")
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

    if vote_test:
        experiment_config.model[0].failure_result = []
        for i in range(len(experiment_config.model)):
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

    # TODO: read this from command line arguments
    attribute = "feature_count"

    print(f"|{attribute}|FAR(%)|FDR(%)|TIA(h)|TIA SD(h)|")
    print("|-------------|------|------|------|---------|")
    for idx, result in enumerate(results):
        val = getattr(experiment_config, attribute)[idx]
        print(
            f"|{val}|{100*result[0]:.2f}|{100*result[1]:.2f}|{result[2]:.1f}|{result[3]:.1f}|"
        )
