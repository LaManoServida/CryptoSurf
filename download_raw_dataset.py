import os
import time

import pandas as pd
from binance import Client

from config import api_key, api_secret, default_dataset_directory


def download_raw_dataset(symbol, interval, start_timestamp_millis, end_timestamp_millis=int(time.time() * 1000),
                         dataset_directory=default_dataset_directory):
    """Download the latest candlestick historical data and save it by default to `default_dataset_directory`.

    Args:
        symbol (str): The currency pair to download data for.
        interval (str): Duration of each candlestick. Must be one of the predefined interval choices.
        start_timestamp_millis (int): Start of the time range in Unix timestamp milliseconds.
        end_timestamp_millis (int, optional): End of the time range in Unix timestamp milliseconds.
        dataset_directory (str, optional): Destination directory for the downloaded dataset.
            Defaults to `default_dataset_directory`.

    Returns:
        str: The file path of the downloaded dataset.
    """

    # create client
    client = Client(api_key, api_secret)

    # download candlestick data
    candles = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_timestamp_millis,
                                           end_str=end_timestamp_millis)

    # get columns of interest and set data type
    candles = pd.DataFrame(
        candles,
        columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
    ).drop('ignore', axis=1)

    # save it to csv
    os.makedirs(dataset_directory, exist_ok=True)
    file_name = f'candles({symbol},{interval},{start_timestamp_millis},{end_timestamp_millis}).csv'
    file_path = os.path.join(dataset_directory, file_name)
    candles.to_csv(file_path, index=False)

    return file_path
