import time

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
    ).drop('ignore', axis=1)

    return df
