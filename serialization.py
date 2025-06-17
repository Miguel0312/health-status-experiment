from dataclasses import dataclass
import dataclasses
import toml
from featureSelection import FeatureSelectionAlgorithm
from preprocess import HealthStatusAlgorithm
import bpnn
import torch


@dataclass
class ExperimentConfig:
    model_type: bpnn.Model = bpnn.Model.Undefined
    model: list[bpnn.FailureDetectionNN] = dataclasses.field(default_factory=list)
    optimizer: list[torch.optim.Optimizer] = dataclasses.field(default_factory=list)
    loss_fn: list[torch.nn.Module] = dataclasses.field(default_factory=list)

    seed: list[int] = dataclasses.field(default_factory=list)
    data_file: list[str] = dataclasses.field(default_factory=list)
    number_of_failing_samples: list[int] = dataclasses.field(default_factory=list)

    change_rate_interval: list[int] = dataclasses.field(default_factory=list)
    feature_count: list[int] = dataclasses.field(default_factory=list)
    feature_selection_algorithm: list[FeatureSelectionAlgorithm] = dataclasses.field(
        default_factory=list
    )
    health_status_algorithm: list[HealthStatusAlgorithm] = dataclasses.field(
        default_factory=list
    )
    good_bad_ratio: float = dataclasses.field(default_factory=list)

    health_status_count: list[int] = dataclasses.field(default_factory=list)
    hidden_nodes: list[int] = dataclasses.field(default_factory=list)
    epoch_count: list[int] = dataclasses.field(default_factory=list)
    learning_rate: list[float] = dataclasses.field(default_factory=list)
    lookback: list[int] = dataclasses.field(default_factory=list)

    vote_count: list[int] = dataclasses.field(default_factory=list)
    vote_threshold: list[float] = dataclasses.field(default_factory=list)

    def print_experiment(self, i) -> str:
        res: str = "{"

        for field in dataclasses.fields(self):
            if field.name in ["model", "optimizer", "loss_fn"]:
                continue

            attr = getattr(self, field.name)
            if type(getattr(self, field.name)) is list:
                attr = attr[i]
            if type(attr) is not int and type(attr) is not float:
                attr = '"' + str(attr) + '"'

            res += f'"{field.name}": {attr},\n'

        res += "}"

        return res


def process_field(field, length):
    if type(field) is list:
        if len(field) != length:
            raise ValueError("All lists must have the same length")
        return field
    else:
        return [field] * length


def load_experiment(file_name: str) -> ExperimentConfig:
    with open(file_name, "r") as f:
        experiment_description = toml.load(f)

    maxi = 1
    for table in experiment_description.values():
        for field in table.values():
            if type(field) is list:
                maxi = max(maxi, len(field))

    config: ExperimentConfig = ExperimentConfig()

    config.seed = process_field(experiment_description["dataset"]["seed"], maxi)
    config.data_file = process_field(
        experiment_description["dataset"]["data_file"], maxi
    )
    config.number_of_failing_samples = process_field(
        experiment_description["dataset"]["number_of_failing_samples"], maxi
    )

    config.change_rate_interval = process_field(
        experiment_description["preprocessing"]["change_rate_interval"], maxi
    )
    config.feature_count = process_field(
        experiment_description["preprocessing"]["feature_count"], maxi
    )
    # TODO: since these two take enums, it should be a little different when there are multiple values
    config.feature_selection_algorithm = process_field(
        FeatureSelectionAlgorithm[
            experiment_description["preprocessing"][
                "feature_selection_algorithm"
            ].upper()
        ],
        maxi,
    )
    config.health_status_algorithm = process_field(
        HealthStatusAlgorithm[
            experiment_description["preprocessing"]["health_status_algorithm"].upper()
        ],
        maxi,
    )
    config.good_bad_ratio = process_field(
        experiment_description["preprocessing"]["good_bad_ratio"], maxi
    )

    config.health_status_count = process_field(
        experiment_description["model"]["health_status_count"], maxi
    )
    config.hidden_nodes = process_field(
        experiment_description["model"]["hidden_nodes"], maxi
    )
    config.epoch_count = process_field(
        experiment_description["model"]["epoch_count"], maxi
    )
    config.learning_rate = process_field(
        experiment_description["model"]["learning_rate"], maxi
    )
    # Lookback is optional
    if "lookback" in experiment_description["model"]:
        config.lookback = process_field(experiment_description["model"]["lookback"], maxi)

    config.vote_count = process_field(
        experiment_description["vote"]["vote_count"], maxi
    )
    config.vote_threshold = process_field(
        experiment_description["vote"]["vote_threshold"], maxi
    )

    config.model_type = bpnn.Model[experiment_description["model"]["model"]]

    for i in range(maxi):
        match config.model_type:
            case bpnn.Model.BinaryBPNN:
                config.model.append(
                    bpnn.BinaryBPNN(config.feature_count[i], config.hidden_nodes[i])
                )
            case bpnn.Model.MultiLevelBPNN:
                config.model.append(
                    bpnn.MultiLevelBPNN(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )
            case bpnn.Model.BinaryRNN:
                config.model.append(
                    bpnn.BinaryRNN(config.feature_count[i], config.hidden_nodes[i])
                )
            case bpnn.Model.MultiLevelRNN:
                config.model.append(
                    bpnn.MultiLevelRNN(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )
            case bpnn.Model.BinaryLSTM:
                config.model.append(
                    bpnn.BinaryLSTM(config.feature_count[i], config.hidden_nodes[i])
                )
            case bpnn.Model.MultiLevelLSTM:
                config.model.append(
                    bpnn.MultiLevelLSTM(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )

    if not config.model:
        raise ValueError("Invalid model type")

    lr_decay = process_field(experiment_description["model"]["lr_decay_interval"], maxi)
    evaluate_interval = process_field(
        experiment_description["model"]["evaluate_interval"], maxi
    )

    for idx, model in enumerate(config.model):
        if not model.validateDescription():
            raise ValueError(
                "It is not possible to create a odel with the given parameters"
            )

        if (
            model.description & bpnn.NNDescription.BINARY
        ) and config.health_status_count[idx] != 2:
            raise ValueError(
                "When training a binary model, the number of classes must be equal to 2"
            )

        if (
            model.description & bpnn.NNDescription.MULTILEVEL
        ) and config.health_status_count[idx] < 3:
            raise ValueError(
                "When training a multi level model, the number of classes must be at least 3"
            )

        model.settings.lr_decay_interval = lr_decay[idx]
        model.settings.evaluate_interval = evaluate_interval[idx]
        model.settings.lookback = config.lookback[idx]

    for i in range(maxi):
        # TODO: try with the Adam optimizer
        config.optimizer.append(
            torch.optim.SGD(
                config.model[i].parameters(),
                lr=config.learning_rate[i],
                weight_decay=1e-7,
            )
        )

    if config.model[0].description & bpnn.NNDescription.BINARY:
        config.loss_fn = [torch.nn.BCELoss()] * maxi
    else:
        config.loss_fn = [torch.nn.CrossEntropyLoss()] * maxi

    return config
