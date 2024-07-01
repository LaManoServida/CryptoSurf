import time

import numpy as np
import pandas as pd
from binance import Client

from config import api_key, api_secret


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
    ).drop('ignore', axis=1).apply(pd.to_numeric)

    return df


def calculate_class_up(df, forecast_horizon, trading_fee_percentage, forecast_gap=0):
    """
    Calculate and add to the dataset a boolean class "up", which is true if any point in the forecast horizon goes up
    with respect to the "close" value, taking buying and selling fees into account.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        forecast_horizon (int): Size of the forecast horizon used to compute the "up" class.
        trading_fee_percentage (float): Fee as a percentage of the asset purchased, used in calculations.
        forecast_gap (int, optional): Number of time steps between the latest "close" value and the forecast horizon.
            Defaults to 0.

    Returns:
        pandas.DataFrame: The input DataFrame with the added "up" column.
    """

    # get the index of 'close' column
    close_index = df.columns.tolist().index('close')

    # get its numpy array representation
    df_array = df.values

    # calculate profit thresholds for all rows in advance
    profit_thresholds = df_array[:, close_index] / (1 - trading_fee_percentage / 100) ** 2

    # iterate over the dataset
    up_list = []
    for i in range(0, len(df) - (forecast_gap + forecast_horizon)):
        forecast_window = df_array[i + 1 + forecast_gap:i + 1 + forecast_gap + forecast_horizon, close_index]
        up_list.append(np.any(forecast_window > profit_thresholds[i]))

    # pad with NaNs
    up_list += [np.nan] * (forecast_gap + forecast_horizon)

    # add the new column
    df['up'] = up_list

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
