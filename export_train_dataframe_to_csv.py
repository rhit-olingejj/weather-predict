import pandas as pd

"""
exports to csv. 
train_dataset is the Train dataset to be exported
returns a string whether the input is "not a dataframe" or "success"
"""
def train_data_to_csv(train_dataset):
    if (isinstance(train_dataset, pd.DataFrame) == False):
        return "not a dataframe"

    train_dataset.to_csv("train_dataset.csv", index=False)

    return "success"