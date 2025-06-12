import random
import pandas as pd
import preprocess

def train_test(
    good_hard_drives: pd.DataFrame,
    bad_hard_drives: pd.DataFrame,
    good_bad_ratio: float = 1,
    bad_samples: int = 12
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:

    serial_number_bad: list[int] = list(bad_hard_drives["serial-number"].unique())
    serial_number_good: list[int] = list(good_hard_drives["serial-number"].unique())

    serial_number_bad_sample: list[int] = random.sample(
        serial_number_bad, int(0.7 * len(serial_number_bad))
    )
    bad_train: pd.DataFrame = bad_hard_drives[
        bad_hard_drives["serial-number"].isin(serial_number_bad_sample)
    ]
    bad_train = preprocess.getLastSamples(bad_train, bad_samples)

    bad_test: pd.DataFrame = bad_hard_drives[
        ~bad_hard_drives["serial-number"].isin(serial_number_bad_sample)
    ]

    serial_number_good_train: list[int] = random.sample(
        serial_number_good, int(0.7 * len(serial_number_good))
    )
    serial_number_good_test: list[int] = list(
        set(serial_number_good) - set(serial_number_good_train)
    )

    good_train: pd.DataFrame = good_hard_drives[
        good_hard_drives["serial-number"].isin(serial_number_good_train)
    ]
    good_train = good_train.sample(n = int(good_bad_ratio * len(bad_train)))
    good_test: pd.DataFrame = good_hard_drives[
        good_hard_drives["serial-number"].isin(serial_number_good_test)
    ]

    df: pd.DataFrame = pd.concat([bad_train, good_train])
    df = df.drop(["Drive Status"], axis=1)

    y_train: pd.Series[int] = df.pop("Health Status")
    x_train: pd.DataFrame = df

    return x_train, y_train, good_test, bad_test
