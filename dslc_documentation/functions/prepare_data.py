import pandas as pd
import numpy as np

def prep_data(data: pd.DataFrame) -> pd.DataFrame:
    for c in data.columns:
        if data[c].max() == 9 and data[c].min() == 1:
            data[c] = data[c].replace(1, 0)
            data[c] = data[c].replace(2, 1)
            data[c] = np.where(data[c] != 0 or data[c] != 1, data[c], np.nan)
    return data