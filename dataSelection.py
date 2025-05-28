import random
import pandas as pd


def train_test(
    good_hard_drives: pd.DataFrame,
    bad_hard_drives: pd.DataFrame,
    good_bad_ratio: float = 1,
    ratioOverSerialNumbers: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    ratio = bad_hard_drives.size / good_hard_drives.size

    serial_number_bad: list[int] = list(bad_hard_drives["serial-number"].unique())
    serial_number_good: list[int] = list(good_hard_drives["serial-number"].unique())

    if ratioOverSerialNumbers:
        ratio = len(serial_number_bad) / len(serial_number_good)

    serial_number_bad_sample: list[int] = random.sample(
        serial_number_bad, int(0.8 * len(serial_number_bad))
    )
    bad_train: pd.DataFrame = bad_hard_drives[
        bad_hard_drives["serial-number"].isin(serial_number_bad_sample)
    ]
    bad_test: pd.DataFrame = bad_hard_drives[
        ~bad_hard_drives["serial-number"].isin(serial_number_bad_sample)
    ]

    serial_number_good_train: list[int] = random.sample(
        serial_number_good, int(0.8 * ratio * good_bad_ratio * len(serial_number_good))
    )
    serial_number_good_test: list[int] = list(
        set(serial_number_good) - set(serial_number_good_train)
    )
    good_train: pd.DataFrame = good_hard_drives[
        good_hard_drives["serial-number"].isin(serial_number_good_train)
    ]
    good_test: pd.DataFrame = good_hard_drives[
        good_hard_drives["serial-number"].isin(serial_number_good_test)
    ]

    df: pd.DataFrame = pd.concat([good_train, bad_train])
    df = df.drop(["Drive Status"], axis=1)

    y_train: pd.Series[int] = df.pop("Health Status")
    x_train: pd.DataFrame = df

    return x_train, y_train, good_test, bad_test
