import pytest
import Train_validate_test_split from ./train_validate_test_split.py

def test_split_sizes():
    X = {
        col1: [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
        col2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    }
    X_size = X.size
    X_train_validate_size = X_size * 0.8
    X_test_size = X_size * 0.2

    X_train_size = X_train_validate_size * 0.8
    X_validate_size = X_train_validate_size * 0.2

    y = {
        col3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    }
    y_size = y.size
    
    y_train_validate_size = y_size * 0.8
    y_test_size = y_size * 0.2

    y_train_size = y_train_validate_size * 0.8
    y_validate_size = y_train_validate_size * 0.2

    X_train, X_validate, X_test, y_train, y_validate, y_test = Train_validate_test_split(X, y)
    
    assert X_train.size == X_train_size
    assert y_train.size == y_train_size

    assert X_validate.size == X_validate_size
    assert y_validate.size == y_validate_size

    assert X_test.size == X_test_size
    assert y_test.size == y_test_size
