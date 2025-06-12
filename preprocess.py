import numpy as np
import numpy.typing as npt
import math
import pandas as pd
from enum import Enum
from typing import Callable
from collections.abc import Hashable


def computeChangeRates(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """
        For each attribute, adds a column to df that computes df[id][t] - df[id][t-interval]
    Deletes the first interval samples of each id, since the change rate can't be computed for them
    """
    # Remove the serial number
    ratesColumns: list[str] = list(df.columns)[2:]
    titles: list[str] = [column + " Change Rate" for column in ratesColumns]
    for title in titles:
        df[title] = [None for i in range(len(df))]

    for idx, column in enumerate(ratesColumns):
        tmpValues: list[float] = list(df[column])
        tmpValues = [np.nan] * interval + tmpValues[:-interval]
        dif: npt.NDArray[np.float64] = np.subtract(df[column], tmpValues)
        df[titles[idx]] = dif

    serial_numbers_shifted: list[int] = list(df["serial-number"])
    serial_numbers_shifted = [-1] * interval + serial_numbers_shifted[:-interval]
    df["serial-numbers-shifted"] = serial_numbers_shifted
    df = df.loc[df["serial-number"] == df["serial-numbers-shifted"]]
    df = df.drop(columns=["serial-numbers-shifted"])

    return df


def getLastSamples(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Returns a dataframe with only the N last samples with each serial-number
    """
    serialNumbers: npt.NDArray[np.int32] = df["serial-number"].unique()
    toKeep: list[Hashable] = []
    for serialNumber in serialNumbers:
        indices: pd.DataFrame = df[df["serial-number"] == serialNumber].index[-N:]
        for index in indices:
            toKeep.append(index)

    return df.loc[toKeep]


class HealthStatusAlgorithm(Enum):
    LINEAR = 1


def LinearAlgorithm(mini: int, maxi: int, i: int, n: int) -> int:
    """
    Linearly map the values [0,n-1] to [maxi, mini]
    """
    return maxi - math.floor((maxi - (mini - 1)) * i / n)


def addHealthStatus(
    df: pd.DataFrame, good: bool, algorithm: HealthStatusAlgorithm, maxLevel: int
) -> pd.DataFrame:
    """
          A column with a score in [0,maxLevel] is given to each sample
    If good is set, it is always equal to maxLevel
    Else the algorithm is used to give a score in [0,maxLevel-1]
    """
    if good:
        df = df.assign(**{"Health Status": [maxLevel for i in range(len(df))]})
        return df

    func: Callable[[int, int, int, int], int] | None = None

    match algorithm:
        case HealthStatusAlgorithm.LINEAR:
            func = LinearAlgorithm

    serialNumbers: npt.NDArray[np.int32] = df["serial-number"].unique()
    healthStatusValues: list[int] = []
    for serialNumber in serialNumbers:
        cnt: int = len(df[df["serial-number"] == serialNumber])
        newValues: list[int] = [func(0, maxLevel - 1, i, cnt) for i in range(cnt)]
        healthStatusValues = healthStatusValues + newValues

    df = df.assign(**{"Health Status": healthStatusValues})
    return df
