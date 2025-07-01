from enum import IntFlag, auto
import torch
import numpy as np
import torch.nn as nn
import random
import pandas as pd
import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralNetworks import FailureDetectionNN


class NNDescription(IntFlag):
    BINARY = auto()
    MULTILEVEL = auto()
    TEMPORAL = auto()
    UNIQUE = auto()
    LSTM = auto()
    RNN = auto()
    BP = auto()


def train(
    model: "FailureDetectionNN",
    epochs: int,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_good: pd.DataFrame,
    test_bad: pd.DataFrame,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    voteCount: int,
) -> None:
    if model.description & NNDescription.BP:
        _train_bp(
            model,
            epochs,
            train_x,
            train_y,
            test_good,
            test_bad,
            loss_fn,
            optimizer,
            voteCount,
        )
    elif model.description & NNDescription.TEMPORAL:
        _train_temporal(
            model,
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
    model: "FailureDetectionNN",
    data_good: pd.DataFrame,
    data_bad: pd.DataFrame,
    voteCount: int,
    ratio: float = 0.5,
) -> None:
    model.eval()

    (far, _, _) = _evaluate_group(model, data_good, voteCount, ratio, 1)
    far = 1 - far
    (fdr, tia, stdDev) = _evaluate_group(model, data_bad, voteCount, ratio, 0, True)
    model.failure_result.append((far, fdr, tia, stdDev))

    print(
        f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%, TIA: {tia:.3f}h, TIA Std Dev: {stdDev:.3f}"
    )


# TODO: transform target into an enum
def _evaluate_group(
    model: "FailureDetectionNN",
    data: pd.DataFrame,
    voteCount: int,
    ratio: float,
    target: int,
    verbose: bool = False,
) -> float:
    with torch.no_grad():
        serialNumbers: npt.NDArray[np.int64] = data["serial-number"].unique()
        count: int = len(serialNumbers)
        X: pd.DataFrame = data.drop(
            columns=["Health Status", "Drive Status", "serial-number"], axis=1
        )
        correct: int = 0
        tia: list[int] = []
        for serialNumber in serialNumbers:
            # indices: list[int] = list(
            #     data[data["serial-number"] == serialNumber].index[-voteCount:]
            # )
            # X_test: torch.Tensor = torch.tensor(
            #     X.loc[indices].values, dtype=torch.float32
            # )
            indices: list[int] = list(data[data["serial-number"] == serialNumber].index)
            X_test: torch.Tensor = torch.tensor(
                X.loc[indices].values, dtype=torch.float32
            )

            result = 1

            for i in range(0, len(X_test) - voteCount):
                candidates = X_test[i : i + voteCount]
                pred = _vote(model, candidates, ratio)
                # if first:
                #     print(serialNumber, candidates, pred)
                if pred == 0:
                    tia.append(len(X_test) - i + voteCount)
                    result = 0
                    break

            if result == target:
                correct += 1

    return (correct / count, np.mean(tia), np.std(tia))


def _vote(model: "FailureDetectionNN", X_values: torch.Tensor, ratio: float) -> int:
    """
    X_values correspond to a sequence of consecutive samples to a given hard drive
    The function returns 0 (the HD is considered as failing) if more than ratio of the samples are considered as failing, else it returns 1
    """
    if model.description & NNDescription.BINARY:
        return _vote_binary(model, X_values, ratio)
    elif model.description & NNDescription.MULTILEVEL:
        return _vote_multilevel(model, X_values, ratio)

    return 0


def _train_bp(
    model: "FailureDetectionNN",
    epochs: int,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_good: pd.DataFrame,
    test_bad: pd.DataFrame,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    voteCount: int,
):
    x_df: pd.DataFrame = train_x.drop(["serial-number"], axis=1)
    x: torch.Tensor = torch.tensor(x_df.values, dtype=torch.float32)

    model.validateDescription()

    # TODO: try a binary model in which the scores are 0.1 and 0.9 for the bad and good examples
    # Since the output value is normalized from 0 to 1, it gives the model ore space
    if model.description & NNDescription.BINARY:
        y: torch.Tensor = torch.tensor(train_y.values, dtype=torch.float32)
        # y.apply_(lambda x: (0.1 + 0.8*x))
        # print(y)
    elif model.description & NNDescription.MULTILEVEL:
        y = torch.tensor(train_y.values, dtype=torch.int64)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs: torch.Tensor = model(x).squeeze()
        loss: torch.Tensor = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        model.loss.append(loss.item())
        if (epoch + 1) % model.settings.lr_decay_interval == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2
        if (epoch + 1) % model.settings.evaluate_interval == 0:
            model.evaluate(test_good, test_bad, voteCount)


def _train_temporal(
    model: "FailureDetectionNN",
    epochs: int,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_good: pd.DataFrame,
    test_bad: pd.DataFrame,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    voteCount: int,
) -> None:
    serialNumbers: npt.NDArray[np.int64] = train_x["serial-number"].unique()
    if model.description & NNDescription.BINARY:
        y: torch.Tensor = torch.tensor(train_y.values, dtype=torch.float32)
    elif model.description & NNDescription.MULTILEVEL:
        y = torch.tensor(train_y.values, dtype=torch.int64)

    batches: list[torch.Tensor] = []
    answer: list[torch.types.Number] = []
    index: int = 0
    lookback: int = model.settings.lookback

    # TODO: test a voting algorithm in which the first sample has weight 1, the second 2, etc.
    # Since there is more lookback for the later samples, there is a higher chance this is correct
    for serialNumber in serialNumbers:
        hd_data: pd.DataFrame = train_x[train_x["serial-number"] == serialNumber]
        hd_data = hd_data.drop(["serial-number"], axis=1)
        hd_data_tensor = torch.tensor(hd_data.values, dtype=torch.float32)
        i: int = len(hd_data) - lookback - 1
        batch = hd_data_tensor[i : i + lookback]
        if len(batch) == lookback:
            batches.append(batch)
            answer.append(y[index + i + 1 : index + lookback + i + 1])
            # print(len(batches[-1]))
            # print(batches[-1])
            # exit(0)

        index += len(hd_data)

    # Need to shuffle the batches because backpropagation is done after each batch
    # There may be errors if the model is trained with a long sequence of samples with the same output
    # TODO: maybe try to equally space the outputs
    # permutation: list[int] = list(range(len(batches)))
    # random.shuffle(permutation)
    # batches = [batches[permutation[i]] for i in range(len(batches))]
    # answer = [answer[permutation[i]] for i in range(len(answer))]

    for epoch in range(epochs):
        model.zero_grad()
        model.net.zero_grad()

        current_loss: float = 0
        for idx, batch in enumerate(batches):
            # print(batch, len(batch))
            output: torch.Tensor = model(batch).squeeze()

            loss: torch.Tensor = loss_fn(output, answer[idx])
            current_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

        current_loss /= len(batches)
        model.loss.append(current_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
        if (epoch + 1) % model.settings.lr_decay_interval == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2
            # print(optimizer.param_groups[0]["lr"])
        if (epoch + 1) % model.settings.evaluate_interval == 0:
            model.evaluate(test_good, test_bad, voteCount)


def _vote_binary(
    model: "FailureDetectionNN", X_values: torch.Tensor, ratio: float
) -> int:
    predictions: torch.Tensor = model(X_values).squeeze()
    predicted_classes: torch.types.Number = (predictions > 0.5).float().sum().item()
    return 1 if predicted_classes >= len(X_values) * (1 - ratio) else 0


def _vote_multilevel(
    model: "FailureDetectionNN", X_values: torch.Tensor, ratio: float
) -> int:
    predictions: torch.Tensor = model(X_values)
    # Algorithm described by Health Status Assessment and Failure Prediction for Hard Drives with Recurrent Neural Networks
    good: int = 0
    for pred in predictions:
        pred = nn.Softmax(dim=0)(pred)
        if pred[:-2].sum() < pred[-1]:
            good += 1
    return 1 if good >= len(X_values) * (1 - ratio) else 0
