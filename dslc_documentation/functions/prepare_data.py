import pandas as pd
import numpy as np
import math
from typing import Optional
from sklearn.model_selection import train_test_split

def num_unique_cols(data: pd.DataFrame, cols: str, num_unique) -> pd.DataFrame:
    "Changes all values above the number of unique items that should be in the columns to NAN"
    data = data.copy(deep=True)
    for c in cols:
        unique_values = data[c].unique()
        
        if len(unique_values) > num_unique:
            if num_unique == 2:
                d = {1: 1, 2: 0}
                data[c] = data[c].map(d).fillna(np.nan)
            else:
                min_value = data[c].min()
                d = {min_value + n: n for n in range(num_unique)}
                data[c] = data[c].map(d).fillna(np.nan)

        if data[c].min() > 0:
            data[c] = data[c] - data[c].min()

    return data

def invalid_to_nan(data: pd.DataFrame, cols: str, invalid_values: list) -> pd.DataFrame:
    "Changes invalid values to NAN"
    data = data.copy(deep=True)
    for c in cols:
        data[c] = data[c].replace(invalid_values, np.nan)
    return data

def categories_oneHot(data: pd.DataFrame, cols: list, drop_first: bool, mappings: dict) -> pd.DataFrame:
    "one hot encodes the columns listed"
    data = data.copy()
    for c in cols:
        data[c] = data[c].map(mappings)
    data = pd.get_dummies(data, columns=cols, drop_first=drop_first)
    return data



def split_data(data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, 
    test_size: float = 0.15, random_state: int = 42) -> tuple:
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    - data: pd.DataFrame - The input data to be split.
    - train_size: float - The proportion of the data to include in the training set.
    - val_size: float - The proportion of the data to include in the validation set.
    - test_size: float - The proportion of the data to include in the test set.
    - random_state: int - The seed used by the random number generator.
    
    Returns:
    - tuple: A tuple containing the training, validation, and test sets as DataFrames.
    """
    assert train_size + val_size + test_size == 1, "The sum of train_size, val_size, and test_size must be 1."
    
    # Split the data into training and temp sets
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_state)
    
    # Calculate the proportion of validation and test sizes relative to the temp set
    val_size_relative = val_size / (val_size + test_size)
    
    # Split the temp set into validation and test sets
    val_data, test_data = train_test_split(temp_data, train_size=val_size_relative, random_state=random_state)
    
    return train_data, val_data, test_data