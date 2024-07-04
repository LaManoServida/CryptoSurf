import numpy as np
import pandas as pd

from logger import logger


def apply_hampel_filter(df, column, window_size=15, n_sigmas=3):
    """
    Apply Hampel filter to detect and treat outliers in the specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to apply the filter to.
        window_size (int, optional): Size of the sliding window. Defaults to 15.
        n_sigmas (int, optional): Number of standard deviations to use as threshold. Defaults to 3.
    """

    logger.info(f'Applying Hampel filter to {column}')

    # Calculate the rolling median
    rolling_median = df[column].rolling(window=window_size, center=True).median()

    # Calculate the threshold
    rolling_mad = df[column].rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))))
    threshold = n_sigmas * rolling_mad

    # Identify outliers
    outliers = np.abs(df[column] - rolling_median) > threshold

    # Replace outliers with the rolling median
    df.loc[outliers, column] = rolling_median[outliers]
