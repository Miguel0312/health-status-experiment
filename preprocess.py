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
    DISCRETE = 1
    NON_SATURATED = 2
    CONTINUOUS = 3


def LinearAlgorithm(good: bool, mini: int, maxi: int, i: int, n: int) -> int:
    """
    Linearly map the values [0,n-1] to [maxi, mini]
    """
    if good:
        return maxi - 1
    val = math.floor((n-i-1)*(maxi-1-mini)/n)
    assert(val >= 0 and val <= maxi - 2)
    return val

def ContinuousAlgorithm(good: bool, mini: int, maxi: int, i: int, n: int) -> float:
    """
    Linearly map the values [0,n-1] to [maxi, mini]
    """
    if good:
        return maxi - 1
    return mini + (maxi-2-mini)*(n-1-i) / (n-1)


def NonSaturatedAlgorithm(good: bool, mini: int, maxi: int, i: int, n: int) -> float:
    return 0.9 if good else 0.1


def computeHealthStatus(
    x: pd.DataFrame, y: pd.Series, algorithm: HealthStatusAlgorithm, maxLevel: int
) -> pd.Series:
    """
          A column with a score in [0,maxLevel] is given to each sample
    If good is set, it is always equal to maxLevel
    Else the algorithm is used to give a score in [0,maxLevel-1]
    """

    func: Callable[[bool, int, int, int, int], int] | None = None

    match algorithm:
        case HealthStatusAlgorithm.DISCRETE:
            func = LinearAlgorithm
        case HealthStatusAlgorithm.CONTINUOUS:
            func = ContinuousAlgorithm
        case HealthStatusAlgorithm.NON_SATURATED:
            assert maxLevel == 1
            func = NonSaturatedAlgorithm

    serialNumbers: npt.NDArray[np.int32] = x["serial-number"].unique()
    healthStatusValues: list[int] = []
    for serialNumber in serialNumbers:
        hd_data = x[x["serial-number"] == serialNumber]
        cnt: int = len(hd_data)
        good = y[hd_data.index[0]] > 0 
        newValues: list[int] = [func(good, 0, maxLevel, i, cnt) for i in range(cnt)]
        healthStatusValues = healthStatusValues + newValues

    return pd.Series(healthStatusValues)
