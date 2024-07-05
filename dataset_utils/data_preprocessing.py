import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, StandardScaler

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


def apply_sg_filter(df, column, window_size=51, polynomial_degree=5, mode='nearest'):
    """
    Apply Savitzky-Golay filter to smooth the specified column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to apply the filter to.
        window_size (int, optional): The length of the filter window (i.e., the number of coefficients).
            Must be a positive odd integer. Defaults to 51.
        polynomial_degree (int, optional): The order of the polynomial used to fit the samples.
            Must be less than window_size. Defaults to 5.
        mode (str, optional): Determines how the array borders are handled. Default is 'nearest'.
    """

    logger.info(f'Applying Savitzky-Golay filter to {column}')

    df[column] = savgol_filter(df[column], window_size, polynomial_degree, mode=mode)


def apply_standard_scaler(df, exclude_columns=None):
    logger.info(f'Applying Standard Scaler to all columns except {exclude_columns}')

    if exclude_columns is None:
        exclude_columns = []

    df[df.columns.difference(exclude_columns)] = StandardScaler().fit_transform(
        df[df.columns.difference(exclude_columns)])


def apply_robust_scaler(df, exclude_columns=None):
    logger.info(f'Applying Robust Scaler to all columns except {exclude_columns}')

    if exclude_columns is None:
        exclude_columns = []

    df[df.columns.difference(exclude_columns)] = RobustScaler().fit_transform(
        df[df.columns.difference(exclude_columns)])
