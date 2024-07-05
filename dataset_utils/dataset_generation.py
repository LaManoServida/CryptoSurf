import time

import numpy as np
import pandas as pd
from binance import Client

from config import api_key, api_secret
from logger import logger


def download_raw_dataset(symbol, interval, start_timestamp_millis, end_timestamp_millis=int(time.time() * 1000)):
    """Download the latest candlestick historical data and return it as a DataFrame.

    Args:
        symbol (str): The currency pair to download data for.
        interval (str): Duration of each candlestick. Must be one of the predefined interval choices.
        start_timestamp_millis (int): Start of the time range in Unix timestamp milliseconds.
        end_timestamp_millis (int, optional): End of the time range in Unix timestamp milliseconds.

    Returns:
        pandas.DataFrame: The downloaded dataset as a DataFrame.
    """

    logger.info('Downloading the raw dataset')

    # create client
    client = Client(api_key, api_secret)

    # download candlestick data
    candles = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_timestamp_millis,
                                           end_str=end_timestamp_millis)

    # get columns of interest and set data type
    df = pd.DataFrame(
        candles,
        columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    ).drop(['close_time', 'ignore'], axis=1).apply(pd.to_numeric)

    return df


def transform_into_sliding_windows(df, window_size, stride=1):
    """
    Transform a dataset with class "up" into sliding windows and return the results as numpy arrays.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset with an 'up' column.
        window_size (int): Size of the sliding windows to be created.
        stride (int, optional): Stride of the sliding windows. Defaults to 1.

    Returns:
        tuple: A 3-tuple containing:
            - numpy.ndarray: X windows data
            - numpy.ndarray: up values corresponding to each window
            - list: Column names of the input DataFrame (excluding 'up')
    """

    logger.info('Transforming the dataset into sliding windows')

    # separate the dataframe from the class "up"
    up_series = df['up']
    df = df.drop('up', axis=1)

    # get its numpy array representation
    df_array = df.values

    # initialize the lists
    x_windows_list = []
    up_list = []

    # iterate over the dataset
    for i in range(0, len(df) - (window_size - 1), stride):
        # append new X window
        x_window = df_array[i:i + window_size]
        x_windows_list.append(x_window)

        # append new class "up" value
        up_list.append(up_series[i + window_size - 1])

    return np.array(x_windows_list), np.array(up_list), df.columns.tolist()


def split_dataset(df, training_size, validation_size):
    """
    Split the dataset into training, validation and test sets.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        training_size (float): Percentage of the dataset to be used for training.
        validation_size (float): Percentage of the dataset to be used for validation.

    Returns:
        tuple: A 3-tuple containing:
            - pandas.DataFrame: Training set
            - pandas.DataFrame: Validation set
            - pandas.DataFrame: Test set
    """

    logger.info('Splitting the dataset into training, validation and test sets')

    total_size = len(df)
    train_end = int(total_size * training_size)
    val_end = train_end + int(total_size * validation_size)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df
