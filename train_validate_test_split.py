from sklearn.model_selection import train_test_split
import pandas as pd

""" Splits X (independent variables) and y (dependent variables) into train, validate, and test datasets """
def Train_validate_test_split(X, y):
    dataframe_size = X.size # since X and y are from the same dataframe, the size is the same

    test_df_size = dataframe_size * 0.2
    validation_df_size = test_df_size * 0.3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_df_size, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = validation_df_size, random_size=42)

    return X_train, X_validate, x_test, y_train, y_validate, y_test


train_validate_df_size = dataframe_size * 0.8
X_train_validate = X.iloc[:train_validate_df_size]
X_test = X.iloc[train_validate_df_size:]
y_train_validate = y.iloc[:train_validate_df_size]
y_test = y.iloc[train_validate_df_size:]

train_df_size = train_validate_df_size * 0.8
X_train = X.iloc[:train_df_size]
X_validate = X.iloc[train_df_size:]
y_train = y.iloc[:train_df_size]
y_validate = y.iloc[train_df_size:]