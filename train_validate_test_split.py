from sklearn.model_selection import train_test_split
import pandas as pd

""" Splits X (independent variables) and y (dependent variables) into train, validate, and test datasets """
def Train_validate_test_split(X, y):
    dataframe_size = X.size # since X and y are from the same dataframe, the size is the same

    test_df_size = 0.2 # test data splits off from train + validation dataset, intial df for X_train and y_train
    train_df_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_df_size, random_state=42)
    print(X_train.size)

    validation_df_size = 0.2 # next split from train
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = validation_df_size, random_state=42) # second values for X_train and y_train with only training data

    return X_train, X_validate, X_test, y_train, y_validate, y_test