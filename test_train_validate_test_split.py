import pytest
import pandas as pd
import numpy as np
from train_validate_test_split import Train_validate_test_split

def test_split_success():
    X = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }

    X_df = pd.DataFrame.from_dict(X)
    X_size = X_df.size
    X_train_validate_size = X_size * 0.8
    X_test_size = X_size * 0.2

    X_train_size = X_train_validate_size * 0.8
    X_validate_size = X_train_validate_size * 0.2

    y = {
        "col3": np.zeros(400)
    }
    y_df = pd.DataFrame.from_dict(y)
    y_size = y_df.size
    
    y_train_validate_size = y_size * 0.8
    y_test_size = y_size * 0.2

    y_train_size = y_train_validate_size * 0.8
    y_validate_size = y_train_validate_size * 0.2

    X_train, X_validate, X_test, y_train, y_validate, y_test = Train_validate_test_split(X_df, y_df)
    
    assert X_train.size == X_train_size
    assert y_train.size == y_train_size

    assert X_validate.size == X_validate_size
    assert y_validate.size == y_validate_size

    assert X_test.size == X_test_size
    assert y_test.size == y_test_size

def test_split_X_not_a_dataframe():
    X = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }

    y = {
        "col3": np.zeros(7)
    }
    y_df = pd.DataFrame.from_dict(y)

    with pytest.raises(ValueError):
        Train_validate_test_split(X, y_df)

def test_split_y_not_a_dataframe():
    X = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }
    X_df = pd.DataFrame.from_dict(X)

    y = {
        "col3": np.zeros(7)
    }

    with pytest.raises(ValueError):
        Train_validate_test_split(X_df, y)


def test_split_X_not_splittable_evenly():
    X = {
        "col1": np.zeros(7),
        "col2": np.zeros(7)
    }

    X_df = pd.DataFrame.from_dict(X)
    X_size = X_df.size
    X_train_validate_size = X_size * 0.8
    X_test_size = X_size * 0.2

    X_train_size = X_train_validate_size * 0.8
    X_validate_size = X_train_validate_size * 0.2

    y = {
        "col3": np.zeros(400)
    }
    y_df = pd.DataFrame.from_dict(y)
    y_size = y_df.size
    
    y_train_validate_size = y_size * 0.8
    y_test_size = y_size * 0.2

    y_train_size = y_train_validate_size * 0.8
    y_validate_size = y_train_validate_size * 0.2

    with pytest.raises(ValueError):
        Train_validate_test_split(X_df, y_df)

def test_split_y_not_splittable_evenly():
    X = {
        "col1": np.zeros(400),
        "col2": np.zeros(400)
    }

    X_df = pd.DataFrame.from_dict(X)
    X_size = X_df.size
    X_train_validate_size = X_size * 0.8
    X_test_size = X_size * 0.2

    X_train_size = X_train_validate_size * 0.8
    X_validate_size = X_train_validate_size * 0.2

    y = {
        "col3": np.zeros(7)
    }
    y_df = pd.DataFrame.from_dict(y)
    y_size = y_df.size
    
    y_train_validate_size = y_size * 0.8
    y_test_size = y_size * 0.2

    y_train_size = y_train_validate_size * 0.8
    y_validate_size = y_train_validate_size * 0.2

    with pytest.raises(ValueError):
        Train_validate_test_split(X_df, y_df)
