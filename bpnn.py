import torch.nn as nn
from utils import NNDescription
import utils
from dataclasses import dataclass
import pandas as pd
import torch.optim as optim
import torch
from enum import Enum


class Model(Enum):
    BinaryBPNN = 1
    MultiLevelBPNN = 2
    BinaryRNN = 3
    MultiLevelRNN = 4
    BinaryLSTM = 5
    MultiLevelLSTM = 6
    Undefined = 7


@dataclass
class ModelSettings:
    input_count: int
    hidden_nodes: int
    output_count: int
    lookback: int
    evaluate_interval: int = 10
    lr_decay_interval: int = 100


class FailureDetectionNN(nn.Module):
    def __init__(
        self,
        input_count: int,
        hidden_nodes: int,
        output_count: int,
        lookback: int = 6,
        evaluate_interval: int = 10,
        lr_decay_interval: int = 100,
    ) -> None:
        super(FailureDetectionNN, self).__init__()
        self.settings = ModelSettings(
            input_count,
            hidden_nodes,
            output_count,
            lookback,
            evaluate_interval,
            lr_decay_interval,
        )
        self.description: NNDescription = NNDescription(0)
        self.net: nn.Module
        self.loss: list[float] = []
        self.failure_result: list[tuple[float, float]] = []

    def train_model(
        self,
        epochs: int,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        test_good: pd.DataFrame,
        test_bad: pd.DataFrame,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        voteCount: int,
    ):
        utils.train(
            self,
            epochs,
            train_x,
            train_y,
            test_good,
            test_bad,
            loss_fn,
            optimizer,
            voteCount,
        )

    def evaluate(
        self,
        data_good: pd.DataFrame,
        data_bad: pd.DataFrame,
        voteCount: int,
        ratio: float = 0.5,
    ):
        utils.evaluate(self, data_good, data_bad, voteCount, ratio)

    def validateDescription(self) -> bool:
        if self.description == 0:
            raise ValueError("No description entered for the model")
        isBinary = self.description & NNDescription.BINARY
        isMultiLevel = self.description & NNDescription.MULTILEVEL
        isTemporal = self.description & NNDescription.TEMPORAL
        isUnique = self.description & NNDescription.UNIQUE
        isLSTM = self.description & NNDescription.LSTM
        isRNN = self.description & NNDescription.RNN
        isBP = self.description & NNDescription.BP

        if isBinary and isMultiLevel:
            raise ValueError(
                "The model must be either binary or multilevel, but not both"
            )
        if (not isBinary) and (not isMultiLevel):
            raise ValueError("The model must be one of binary or multilevel")
        if isTemporal and isUnique:
            raise ValueError(
                "The model must be either unique or temporal, but not both"
            )
        if (not isTemporal) and (not isUnique):
            raise ValueError("The model must be one of unique or temporal")
        if isUnique and not isBP:
            raise ValueError("A unique model must be temporal")
        if isTemporal and (not isLSTM and not isRNN):
            raise ValueError("A temporal model must be either an LSTM or an RNN")
        if (isBP | isRNN | isLSTM).bit_count() != 1:
            raise ValueError("A description must be exactly one of BP, RNN and LSTM")

        return True


class BinaryBPNN(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int) -> None:
        super(BinaryBPNN, self).__init__(input_count, hidden_nodes, 1)
        self.description |= (
            NNDescription.BP | NNDescription.BINARY | NNDescription.UNIQUE
        )
        self.net = nn.Sequential(
            nn.Linear(input_count, hidden_nodes),
            # nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiLevelBPNN(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int, output_count: int) -> None:
        super(MultiLevelBPNN, self).__init__(input_count, hidden_nodes, output_count)
        self.description |= (
            NNDescription.BP | NNDescription.MULTILEVEL | NNDescription.UNIQUE
        )
        self.output_count = output_count
        self.net = nn.Sequential(
            nn.Linear(input_count, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, output_count),
            # nn.Sigmoid(),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class BinaryRNN(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int) -> None:
        super(BinaryRNN, self).__init__(input_count, hidden_nodes, 2)

        self.description |= (
            NNDescription.BINARY | NNDescription.TEMPORAL | NNDescription.RNN
        )
        self.net = nn.RNN(input_count, hidden_nodes, nonlinearity="relu")
        self.linear = nn.Linear(hidden_nodes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.net(x)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class MultiLevelRNN(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int, output_count: int) -> None:
        super(MultiLevelRNN, self).__init__(input_count, hidden_nodes, output_count)

        self.description |= (
            NNDescription.MULTILEVEL | NNDescription.TEMPORAL | NNDescription.RNN
        )
        self.net = nn.RNN(input_count, hidden_nodes, nonlinearity="relu")
        self.linear = nn.Linear(hidden_nodes, output_count)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.net(x)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class BinaryLSTM(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int) -> None:
        super(BinaryLSTM, self).__init__(input_count, hidden_nodes, 2)

        self.description |= (
            NNDescription.BINARY | NNDescription.LSTM | NNDescription.TEMPORAL
        )
        self.net = nn.LSTM(input_count, hidden_nodes, batch_first=True)
        self.linear = nn.Linear(hidden_nodes, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.net(x)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class MultiLevelLSTM(FailureDetectionNN):
    def __init__(self, input_count: int, hidden_nodes: int, output_count: int) -> None:
        super(MultiLevelLSTM, self).__init__(input_count, hidden_nodes, output_count)

        self.description |= (
            NNDescription.MULTILEVEL | NNDescription.LSTM | NNDescription.TEMPORAL
        )
        self.net = nn.LSTM(input_count, hidden_nodes, batch_first=True)
        self.linear = nn.Linear(hidden_nodes, output_count)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.net(x)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
