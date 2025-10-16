import numpy as np
import pandas as pd
import pytest


DATE_FEATURE = "issue_d"
N_SAMPLES: int | None = 1000


def sample(df: pd.DataFrame) -> pd.DataFrame:
    if N_SAMPLES:
        out : pd.DataFrame = df.iloc[:N_SAMPLES]
        return out
    
    return df

def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(df[DATE_FEATURE])
    i_train = np.where(
        (pd.to_datetime("2013-01-01") <= t) & (t <= pd.to_datetime("2015-12-31"))
    )[0]
    i_test = np.where(
        (pd.to_datetime("2016-01-01") <= t) & (t <= pd.to_datetime("2017-12-31"))
    )[0]
    out: tuple[pd.DataFrame, pd.DataFrame] = df.iloc[i_train], df.iloc[i_test]
    return out


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return pd.read_csv("./tests/data/lcld_v2.csv")


@pytest.fixture(scope="session")
def train_data(dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(dataset)[0])

@pytest.fixture(scope="session")
def test_data(dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(dataset)[1])
