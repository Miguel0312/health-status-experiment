from dataclasses import dataclass
import dataclasses
import toml
from featureSelection import FeatureSelectionAlgorithm
from preprocess import HealthStatusAlgorithm
import neuralNetworks
import decisionTrees
import torch
import modelBase
import utils


@dataclass
class ExperimentConfig:
    model_type: modelBase.ModelType = modelBase.ModelType.UNDEFINED
    model: list[modelBase.FailureDetectionModel] = dataclasses.field(
        default_factory=list
    )

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

    vote_count: list[int] = dataclasses.field(default_factory=list)
    vote_threshold: list[float] = dataclasses.field(default_factory=list)
    voting_algorithm: list[utils.VotingAlgorithm] = dataclasses.field(
        default_factory=list
    )

    def print_experiment(self, i) -> str:
        res: str = "{"

        for field in dataclasses.fields(self):
            if field.name in ["model", "optimizer", "loss_fn"]:
                continue

            attr = getattr(self, field.name)
            if type(getattr(self, field.name)) is list:
                if i < len(attr):
                    attr = attr[i]
            if type(attr) is not int and type(attr) is not float:
                attr = '"' + str(attr) + '"'

            res += f'"{field.name}": {attr},\n'

        res += "}"

        return res


@dataclass
class NeuralNetworkConfig(ExperimentConfig):
    model_type: modelBase.ModelType = modelBase.ModelType.NN
    nn_type: neuralNetworks.Model = neuralNetworks.Model.Undefined
    optimizer: list[torch.optim.Optimizer] = dataclasses.field(default_factory=list)
    loss_fn: list[torch.nn.Module] = dataclasses.field(default_factory=list)
    hidden_nodes: list[int] = dataclasses.field(default_factory=list)
    epoch_count: list[int] = dataclasses.field(default_factory=list)
    learning_rate: list[float] = dataclasses.field(default_factory=list)
    lookback: list[int] = dataclasses.field(default_factory=list)

@dataclass
class DecisionTreeConfig(ExperimentConfig):
    model_type: modelBase.ModelType = modelBase.ModelType.TREE
    tree_type: decisionTrees.TreeType = decisionTrees.TreeType.UNDEFINED
    criterion: list[decisionTrees.TreeCriterion] = dataclasses.field(
        default_factory=list
    )
    max_depth: list[int] = dataclasses.field(default_factory=list)
    min_samples_leaf: list[int] = dataclasses.field(default_factory=list)


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

    model_type = modelBase.ModelType[experiment_description["model"]["model_type"]]

    match model_type:
        case modelBase.ModelType.NN:
            return _load_neural_network(experiment_description)
        case modelBase.ModelType.TREE:
            return _load_tree(experiment_description)
        case _:
            raise ValueError("No valid value of field model.model_type")


def _load_base(config: ExperimentConfig, experiment_description):
    maxi = 1
    for table in experiment_description.values():
        for field in table.values():
            if type(field) is list:
                maxi = max(maxi, len(field))

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

    if (
        type(experiment_description["preprocessing"]["feature_selection_algorithm"])
        is list
    ):
        config.feature_selection_algorithm = process_field(
            [
                FeatureSelectionAlgorithm[x.upper()]
                for x in experiment_description["preprocessing"][
                    "feature_selection_algorithm"
                ]
            ],
            maxi,
        )
    else:
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

    config.vote_count = process_field(
        experiment_description["vote"]["vote_count"], maxi
    )
    config.vote_threshold = process_field(
        experiment_description["vote"]["vote_threshold"], maxi
    )

    if "voting_algorithm" not in experiment_description["vote"]:
        config.voting_algorithm = [utils.VotingAlgorithm.STANDARD] * maxi
    elif type(experiment_description["vote"]["voting_algorithm"]) is list:
        config.voting_algorithm = process_field(
            [
                utils.VotingAlgorithm[x.upper()]
                for x in experiment_description["vote"]["voting_algorithm"]
            ],
            maxi,
        )
    else:
        config.voting_algorithm = process_field(
            utils.VotingAlgorithm[
                experiment_description["vote"]["voting_algorithm"].upper()
            ],
            maxi,
        )

    return maxi


def _load_neural_network(experiment_description):
    config = NeuralNetworkConfig()

    maxi = _load_base(config, experiment_description)

    config.hidden_nodes = process_field(
        experiment_description["model"]["hidden_nodes"], maxi
    )
    config.epoch_count = process_field(
        experiment_description["model"]["epoch_count"], maxi
    )
    config.learning_rate = process_field(
        experiment_description["model"]["learning_rate"], maxi
    )
    # TODO: if the model is temporal, then it needs a lookback
    if "lookback" in experiment_description["model"]:
        config.lookback = process_field(
            experiment_description["model"]["lookback"], maxi
        )

    config.nn_type = neuralNetworks.Model[experiment_description["model"]["model"]]

    for i in range(maxi):
        match config.nn_type:
            case neuralNetworks.Model.BinaryBPNN:
                config.model.append(
                    neuralNetworks.BinaryBPNN(
                        config.feature_count[i], config.hidden_nodes[i]
                    )
                )
            case neuralNetworks.Model.MultiLevelBPNN:
                config.model.append(
                    neuralNetworks.MultiLevelBPNN(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )
            case neuralNetworks.Model.BinaryRNN:
                config.model.append(
                    neuralNetworks.BinaryRNN(
                        config.feature_count[i], config.hidden_nodes[i]
                    )
                )
            case neuralNetworks.Model.MultiLevelRNN:
                config.model.append(
                    neuralNetworks.MultiLevelRNN(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )
            case neuralNetworks.Model.BinaryLSTM:
                config.model.append(
                    neuralNetworks.BinaryLSTM(
                        config.feature_count[i], config.hidden_nodes[i]
                    )
                )
            case neuralNetworks.Model.MultiLevelLSTM:
                config.model.append(
                    neuralNetworks.MultiLevelLSTM(
                        config.feature_count[i],
                        config.hidden_nodes[i],
                        config.health_status_count[i],
                    )
                )
            case _:
                raise ValueError(
                    "Value of field model.model doesn't match any neural network architecture"
                )

    lr_decay = process_field(experiment_description["model"]["lr_decay_interval"], maxi)
    evaluate_interval = process_field(
        experiment_description["model"]["evaluate_interval"], maxi
    )

    for idx, model in enumerate(config.model):
        if not model.validateDescription():
            raise ValueError(
                "It is not possible to create a model with the given parameters"
            )

        if (
            model.description & neuralNetworks.NNDescription.BINARY
        ) and config.health_status_count[idx] != 2:
            raise ValueError(
                "When training a binary model, the number of classes must be equal to 2"
            )

        if (
            model.description & neuralNetworks.NNDescription.MULTILEVEL
        ) and config.health_status_count[idx] < 3:
            raise ValueError(
                "When training a multi level model, the number of classes must be at least 3"
            )

        model.settings.lr_decay_interval = lr_decay[idx]
        model.settings.evaluate_interval = evaluate_interval[idx]
        if model.description & neuralNetworks.NNDescription.TEMPORAL:
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

    # TODO: adapth this when there are models of different types
    if config.model[0].description & neuralNetworks.NNDescription.BINARY:
        config.loss_fn = [torch.nn.BCELoss()] * maxi
    else:
        config.loss_fn = [torch.nn.CrossEntropyLoss()] * maxi

    return config


def _load_tree(experiment_description):
    config = DecisionTreeConfig()

    maxi = _load_base(config, experiment_description)

    if type(experiment_description["model"]["criterion"]) is list:
        config.criterion = process_field(
            [x.upper() for x in experiment_description["model"]["criterion"]],
            maxi,
        )
    else:
        config.criterion = process_field(
            decisionTrees.TreeCriterion[
                experiment_description["model"]["criterion"].upper()
            ],
            maxi,
        )

    config.max_depth = process_field(experiment_description["model"]["max_depth"], maxi)
    config.min_samples_leaf = process_field(
        experiment_description["model"]["min_samples_leaf"], maxi
    )

    config.tree_type = decisionTrees.TreeType[
        experiment_description["model"]["tree_type"]
    ]

    for i in range(maxi):
        if config.tree_type == decisionTrees.TreeType.CLASSIFICATION:
            config.model.append(
                decisionTrees.ClassificationTree(
                    config.criterion[i], config.max_depth[i], config.min_samples_leaf[i]
                )
            )
        elif config.tree_type == decisionTrees.TreeType.REGRESSION:
            config.model.append(
                decisionTrees.RegressionTree(
                    config.criterion[i], config.max_depth[i], config.min_samples_leaf[i]
                )
            )
        else:
            raise ValueError("Value of tree_type is invalid")

    return config
