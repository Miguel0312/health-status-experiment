from enum import Enum
import numpy as np
import math
import pandas as pd
from typing import Callable


class FeatureSelectionAlgorithm(Enum):
    Z_SCORE = 1


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


def selectFeatures(
    df: pd.DataFrame, algorithm: FeatureSelectionAlgorithm, toKeepCount: int
) -> pd.DataFrame:
    """
    Returns df with only toKeepCount features corresponding to the ones that score the highest according to algorithm
    """
    func: Callable[[list[float], list[float]], float] | None = None
    match algorithm:
        case FeatureSelectionAlgorithm.Z_SCORE:
            func = z_score

    # Remove the serial and status
    columns: list[str] = list(df.columns)[2:]
    good_hard_drives: pd.DataFrame = df[df["Drive Status"] == 1]
    bad_hard_drives: pd.DataFrame = df[df["Drive Status"] == -1]

    results: list[tuple[float, str]] = []

    for col in columns:
        goodSamples: list[float] = list(good_hard_drives[col])
        badSamples: list[float] = list(bad_hard_drives[col])

        results.append((func(goodSamples, badSamples), col))

    results.sort(reverse=True)
    toKeep: list[str] = (
        list(df.columns)[0:2] + [result[1] for result in results][:toKeepCount]
    )

    return df[toKeep]
