from modelBase import FailureDetectionModel
from enum import Enum
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import torch
import numpy as np
import utils

class TreeType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    UNDEFINED = 3


class TreeCriterion(Enum):
    GINI = "gini"
    ENTROPY = "entropy"
    SQUARED_ERROR = "squared_error"


class ClassificationTree(FailureDetectionModel):
    def __init__(self, criterion: TreeCriterion, max_depth: int, min_samples_leaf: int):
        super(ClassificationTree, self).__init__()
        self.criterion: TreeCriterion = criterion
        self.max_depth: int = max_depth
        self.min_samples_leaf: int = min_samples_leaf
        self.tree: DecisionTreeClassifier = DecisionTreeClassifier(
            criterion=self.criterion.value,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def train_model(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_good,
        test_bad,
        voteCount,
        vote_threshold,
        voting_algorithm: utils.VotingAlgorithm
    ):
        x_df: pd.DataFrame = train_x.drop(["serial-number"], axis=1)
        x: torch.Tensor = torch.tensor(x_df.values, dtype=torch.float32)

        y: torch.Tensor = torch.tensor(train_y.values, dtype=torch.float32)

        self.tree.fit(x, y)
        self.evaluate(test_good, test_bad, voteCount)

    def evaluate(
        self,
        data_good: pd.DataFrame,
        data_bad: pd.DataFrame,
        vote_count: int,
        ratio: float = 0.5,
        voting_algorithm: utils.VotingAlgorithm = utils.VotingAlgorithm.STANDARD
    ):
        (far, _, _) = self._evaluate_group(data_good, vote_count, ratio, 1)

        far = 1 - far
        (fdr, tia, stdDev) = self._evaluate_group(data_bad, vote_count, ratio, 0)
        self.failure_result.append((far, fdr, tia, stdDev))

        print(
            f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%, TIA: {tia:.3f}h, TIA Std Dev: {stdDev:.3f}"
        )

    def _evaluate_group(
        self, data: pd.DataFrame, voteCount: int, ratio: float, target: int
    ):
        serialNumbers = data["serial-number"].unique()
        count: int = len(serialNumbers)
        X: pd.DataFrame = data.drop(
            columns=["Drive Status", "serial-number"], axis=1
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
                pred = self._vote(candidates, ratio)
                # if first:
                #     print(serialNumber, candidates, pred)
                if pred == 0:
                    tia.append(len(X_test) - i + voteCount)
                    result = 0
                    break

            if result == target:
                correct += 1

        return (correct / count, np.mean(tia), np.std(tia))

    def _vote(self, x_values: torch.Tensor, ratio: float):
        pred = self.tree.predict(x_values)
        s = sum(pred)

        # TODO: VAT2H is needed when using more than 2 health status and more than 1 vote

        return 1 if s >= len(x_values) * (1 - ratio) else 0


# TODO: refactor the common code with ClassificationTree
class RegressionTree(FailureDetectionModel):
    def __init__(self, criterion: TreeCriterion, max_depth: int, min_samples_leaf: int, health_status_count: int):
        super(RegressionTree, self).__init__()
        self.criterion: TreeCriterion = criterion
        self.max_depth: int = max_depth
        self.min_samples_leaf: int = min_samples_leaf
        self.health_status_count: int = health_status_count
        self.tree: DecisionTreeRegressor = DecisionTreeRegressor(
            criterion=self.criterion.value,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def train_model(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_good,
        test_bad,
        voteCount,
        vote_threshold,
        voting_algorithm: utils.VotingAlgorithm
    ):
        x_df: pd.DataFrame = train_x.drop(["serial-number"], axis=1)
        x: torch.Tensor = torch.tensor(x_df.values, dtype=torch.float32)

        y: torch.Tensor = torch.tensor(train_y.values, dtype=torch.float32)
        self.tree.fit(x, y)
        self.evaluate(test_good, test_bad, voteCount, vote_threshold, voting_algorithm)

    def evaluate(
        self,
        data_good: pd.DataFrame,
        data_bad: pd.DataFrame,
        voteCount: int,
        ratio: float = 0.5,
        voting_algorithm: utils.VotingAlgorithm = utils.VotingAlgorithm.STANDARD
    ):
        (far, _, _) = self._evaluate_group(data_good, voteCount, ratio, 1, voting_algorithm)

        far = 1 - far
        (fdr, tia, stdDev) = self._evaluate_group(data_bad, voteCount, ratio, 0, voting_algorithm)
        self.failure_result.append((far, fdr, tia, stdDev))

        print(
            f"FAR: {100*far:.3f}%, FDR: {100*fdr:.3f}%, TIA: {tia:.3f}h, TIA Std Dev: {stdDev:.3f}"
        )

    def _evaluate_group(
        self, data: pd.DataFrame, voteCount: int, ratio: float, target: int, voting_algorithm: utils.VotingAlgorithm
    ):
        serialNumbers = data["serial-number"].unique()
        count: int = len(serialNumbers)
        X: pd.DataFrame = data.drop(
            columns=["Drive Status", "serial-number"], axis=1
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
                pred = self._vote(candidates, ratio, voting_algorithm)
                # if first:
                #     print(serialNumber, candidates, pred)
                if pred == 0:
                    tia.append(len(X_test) - (i + voteCount))
                    result = 0
                    break

            if result == target:
                correct += 1

        return (correct / count, np.mean(tia), np.std(tia))

    def _vote(self, x_values: torch.Tensor, ratio: float, voting_algorithm: utils.VotingAlgorithm):
        if voting_algorithm == utils.VotingAlgorithm.STANDARD:
            s = sum(self.tree.predict(x_values))
            lim = 1-ratio if self.health_status_count == 2 else self.health_status_count - 2
            return 1 if s >= lim*len(x_values) else 0
        elif voting_algorithm == utils.VotingAlgorithm.VAT2H:
            pred = self.tree.predict(x_values)
            cnt = [0]*self.health_status_count
            for p in pred:
                cnt[round(p)] += 1
            
            if self.health_status_count == 2:
                return 1 if cnt[0] < cnt[1] else 0
            return 1 if cnt[:-2] < cnt[-1] else 0
        else:
            raise(ValueError("Invalid Value for Voting Algorithm"))
