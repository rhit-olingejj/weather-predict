from sklearn.model_selection import train_test_split
import pandas as pd

""" Splits X (independent variables) and y (dependent variables) into train, validate, and test datasets """
def Train_validate_test_split(X, y):
    if (isinstance(X, pd.DataFrame) == False):
        raise ValueError("X is not a dataframe")

    if (isinstance(y, pd.DataFrame) == False):
        raise ValueError("y is not a dataframe")

    # Counted in rows, not cells: .size is rows * columns, so a 65-feature matrix only cleared a % 8 test by luck and 8,985 real rows never could. Eight is the floor at which 64/16/20 still puts at least one row in each split.
    if (len(X) < 8):
        raise ValueError("Cannot evenly split X: fewer than 8 rows")

    if (len(y) < 8):
        raise ValueError("Cannot evenly split y: fewer than 8 rows")

    dataframe_size = len(X) # since X and y are from the same dataframe, the size is the same

    test_df_size = 0.2 # test data splits off from train + validation dataset, intial df for X_train and y_train
    train_df_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_df_size, random_state=42)

    validation_df_size = 0.2 # next split from train
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = validation_df_size, random_state=42) # second values for X_train and y_train with only training data

    return X_train, X_validate, X_test, y_train, y_validate, y_test