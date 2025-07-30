from enum import Enum
import numpy as np
import math
import pandas as pd
import random
import bisect
from scipy.stats import ranksums


class FeatureSelectionAlgorithm(Enum):
    Z_SCORE = 1
    REVERSE_ARRANGEMENT = 2
    RANK_SUM = 3


def z_score(goodSamples: list[float], badSamples: list[float]) -> float:
    """
          Input: two lists with the good and bad samples
    Output: the z-score that measure how different the distribution of the values are between the good and bad samples
    """
    nf: int = len(badSamples)
    ng: int = len(goodSamples)
    mf: float = float(np.average(badSamples))
    mg: float = float(np.average(goodSamples))
    vf: float = float(np.var(badSamples))
    vg: float = float(np.var(goodSamples))

    if vf == 0 and vg == 0:
        return 0

    return math.fabs(mf - mg) / math.sqrt(vf / nf + vg / ng)


def rank_sum(goodSamples: list[float], badSamples: list[float]) -> float:
    u, p = ranksums(goodSamples, badSamples)
    return -p


def positive(samples: list[list[float]]):
    cnt = 0
    samples = list(filter(lambda x: len(x) >= 100, samples))
    samples = [x[-100:] for x in samples]

    # Values from table A.6 from appendix of Random Data Analysis and Measurements Procedures, Bendat and Piersol
    # alpha = 2%
    left = 2083
    right = 2866

    mini = 1000000
    maxi = -1

    for x in samples:
        inv = inversions(x)
        mini = min(inv, mini)
        maxi = max(inv, maxi)
        if inv >= left and inv <= right:
            cnt += 1

    return cnt / len(samples)


def inversions(x: list[float]):
    # Add noise to prevent equalities
    x = [y + random.uniform(-0.000001, 0.000001) for y in x]
    cnt = 0
    elements = []
    for y in x:
        # Find the number of elements bigger than y
        index = bisect.bisect(elements, y)
        cnt += len(elements) - index
        elements.insert(index, y)
    return cnt


def reverse_arrangement(
    goodSamples: list[list[float]], badSamples: list[list[float]]
) -> float:
    # Sources: Hard drive failure prediction using non-parametric statistical methods
    # Machine Learning Methods for Predicting Failure in Hard Drives: A Multiple-Instance Application

    good_ratio = positive(goodSamples)
    bad_ratio = positive(badSamples)

    return good_ratio, bad_ratio


def selectFeatures(
    df: pd.DataFrame, algorithm: FeatureSelectionAlgorithm, toKeepCount: int
) -> pd.DataFrame:
    """
    Returns df with only toKeepCount features corresponding to the ones that score the highest according to algorithm
    """
    func = None
    match algorithm:
        case FeatureSelectionAlgorithm.Z_SCORE:
            func = z_score
        case FeatureSelectionAlgorithm.REVERSE_ARRANGEMENT:
            func = reverse_arrangement
        case FeatureSelectionAlgorithm.RANK_SUM:
            func = rank_sum

    # Remove the serial and status
    columns: list[str] = list(df.columns)[2:]
    good_hard_drives: pd.DataFrame = df[df["Drive Status"] == 1]
    bad_hard_drives: pd.DataFrame = df[df["Drive Status"] == -1]

    results = []

    if (
        algorithm == FeatureSelectionAlgorithm.Z_SCORE
        or algorithm == FeatureSelectionAlgorithm.RANK_SUM
    ):
        for col in columns:
            goodSamples: list[float] = list(good_hard_drives[col])
            badSamples: list[float] = list(bad_hard_drives[col])

            results.append((func(goodSamples, badSamples), col))

        results.sort(reverse=True)
        toKeep: list[str] = (
            list(df.columns)[0:2] + [result[1] for result in results][:toKeepCount]
        )

        return df[toKeep]
    elif algorithm == FeatureSelectionAlgorithm.REVERSE_ARRANGEMENT:
        goodSerialNumbers = good_hard_drives["serial-number"].unique()
        badSerialNumbers = bad_hard_drives["serial-number"].unique()

        goodSamplesProcessed = []
        badSamplesProcessed = []
        for _ in range(len(columns)):
            goodSamplesProcessed.append([])
            badSamplesProcessed.append([])

        for serialNumber in goodSerialNumbers:
            hwDataSet = good_hard_drives[
                good_hard_drives["serial-number"] == serialNumber
            ]
            for idx, col in enumerate(columns):
                goodSamplesProcessed[idx].append(list(hwDataSet[col]))
        for serialNumber in badSerialNumbers:
            hwDataSet = bad_hard_drives[
                bad_hard_drives["serial-number"] == serialNumber
            ]
            for idx, col in enumerate(columns):
                badSamplesProcessed[idx].append(list(hwDataSet[col]))

        for idx, col in enumerate(columns):
            results.append(
                (func(goodSamplesProcessed[idx], badSamplesProcessed[idx]), col)
            )

        results.sort(key=lambda x: abs(x[0][0] - x[0][1]), reverse=True)

        toKeep: list[str] = (
            list(df.columns)[0:2] + [result[1] for result in results][:toKeepCount]
        )

        return df[toKeep]
