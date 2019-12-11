import numpy as np
import pandas as pd


def ecdf_vals(data):
    """Return x-y coords for point-based ECDF for inputted data"""
    # Get sorted data with number of instances of each data point
    value, count = np.unique(data, return_counts=True)

    # Calculate ECDF y coords with ratio of cumulative sum to total data
    ecdf = np.cumsum(count) / len(data)

    df = pd.DataFrame({"x": value, "y": ecdf})
    return df