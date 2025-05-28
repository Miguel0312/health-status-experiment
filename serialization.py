from dataclasses import dataclass
import dataclasses
import toml
from featureSelection import FeatureSelectionAlgorithm
from preprocess import HealthStatusAlgorithm
import bpnn
import torch

file_name = "experiments/test.toml"


@dataclass
class ExperimentConfig:
    model_type: bpnn.Model = bpnn.Model.Undefined
    model: bpnn.FailureDetectionNN | None = None
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: torch.nn.Module | None = None

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

    def to_string(self) -> str:
        res: str = "{"

        for field in dataclasses.fields(self):
            res += f"{field.name}: {getattr(self, field.name)},\n"

        res += "}"

        return res


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

    config.model_type = bpnn.Model[experiment_description["model"]["model"]]
    match config.model_type:
        case bpnn.Model.BinaryBPNN:
            config.model = bpnn.BinaryBPNN(config.feature_count, config.hidden_nodes)
        case bpnn.Model.MultiLevelBPNN:
            config.model = bpnn.MultiLevelBPNN(
                config.feature_count, config.hidden_nodes, config.health_status_count
            )
        case bpnn.Model.BinaryRNN:
            config.model = bpnn.BinaryRNN(config.feature_count, config.hidden_nodes)
        case bpnn.Model.MultiLevelRNN:
            config.model = bpnn.MultiLevelRNN(
                config.feature_count, config.hidden_nodes, config.health_status_count
            )
        case bpnn.Model.BinaryLSTM:
            config.model = bpnn.BinaryLSTM(config.feature_count, config.hidden_nodes)
        case bpnn.Model.MultiLevelLSTM:
            config.model = bpnn.MultiLevelLSTM(
                config.feature_count, config.hidden_nodes, config.health_status_count
            )

    if config.model is None or not config.model.validateDescription():
        raise ValueError("Could not initialize model with the given configuration")

    if (
        config.model.description & bpnn.NNDescription.BINARY
    ) and config.health_status_count != 2:
        raise ValueError(
            "When training a binary model, the number of classes must be equal to 2"
        )

    if (
        config.model.description & bpnn.NNDescription.MULTILEVEL
    ) and config.health_status_count < 3:
        raise ValueError(
            "When training a multi level model, the number of classes must be at least 3"
        )

    config.model.settings.lr_decay_interval = experiment_description["model"][
        "lr_decay_interval"
    ]
    config.model.settings.evaluate_interval = experiment_description["model"][
        "evaluate_interval"
    ]

    # TODO: try with the Adam optimizer
    config.optimizer = torch.optim.SGD(
        config.model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-7,
    )

    if config.model.description & bpnn.NNDescription.BINARY:
        config.loss_fn = torch.nn.BCELoss()
    else:
        config.loss_fn = torch.nn.CrossEntropyLoss()

    return config
