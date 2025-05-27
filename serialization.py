from dataclasses import dataclass
import toml
from featureSelection import FeatureSelectionAlgorithm
from preprocess import HealthStatusAlgorithm

file_name = "experiments/test.toml"


@dataclass
class ExperimentConfig:
    seed: int = 0
    data_file: str = ""
    number_of_failing_samples: int = 0

    change_rate_interval: int = 0
    feature_count: int = 0
    feature_selection_algorithm: FeatureSelectionAlgorithm = (
        FeatureSelectionAlgorithm.Z_SCORE
    )
    health_status_algorithm: HealthStatusAlgorithm = HealthStatusAlgorithm.LINEAR
    good_bad_ratio: float = 1.0

    health_status_count: int = 6
    hidden_nodes: int = 10
    epoch_count: int = 400
    learning_rate: float = 0.1

    vote_count: int = 1
    vote_threshold: float = 0.5


def load_experiment(file_name: str) -> ExperimentConfig:
    with open(file_name, "r") as f:
        experiment_description = toml.load(f)

    config: ExperimentConfig = ExperimentConfig()

    config.seed = experiment_description["dataset"]["seed"]
    config.data_file = experiment_description["dataset"]["data_file"]
    config.number_of_failing_samples = experiment_description["dataset"][
        "number_of_failing_samples"
    ]

    config.change_rate_interval = experiment_description["preprocessing"][
        "change_rate_interval"
    ]
    config.feature_count = experiment_description["preprocessing"]["feature_count"]
    config.feature_selection_algorithm = FeatureSelectionAlgorithm[
        experiment_description["preprocessing"]["feature_selection_algorithm"].upper()
    ]
    config.health_status_algorithm = HealthStatusAlgorithm[
        experiment_description["preprocessing"]["health_status_algorithm"].upper()
    ]
    config.good_bad_ratio = experiment_description["preprocessing"]["good_bad_ratio"]

    config.health_status_count = experiment_description["model"]["health_status_count"]
    config.hidden_nodes = experiment_description["model"]["hidden_nodes"]
    config.epoch_count = experiment_description["model"]["epoch_count"]
    config.learning_rate = experiment_description["model"]["learning_rate"]

    config.vote_count = experiment_description["vote"]["vote_count"]
    config.vote_threshold = experiment_description["vote"]["vote_threshold"]

    return config


print(load_experiment(file_name))
