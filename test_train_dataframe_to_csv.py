import pytest
import pandas as pd
import numpy as np
from export_train_dataframe_to_csv import train_data_to_csv

def test_train_data_to_csv_success():
    train_dataset = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }

    train_dataset_df = pd.DataFrame.from_dict(train_dataset)

    result = train_data_to_csv(train_dataset_df)

    assert result == "success"

def test_not_a_dataframe():
    train_dataset = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }

    result = train_data_to_csv(train_dataset)

    assert result == "not a dataframe"