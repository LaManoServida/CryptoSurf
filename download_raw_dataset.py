import argparse
import os
import time

import pandas as pd
from binance import Client

from config import api_key, api_secret, default_dataset_directory


def download_raw_dataset(symbol, interval, start_timestamp_millis, end_timestamp_millis=int(time.time() * 1000),
                         dataset_directory=default_dataset_directory):
    """Download the latest candlestick historical data and save it by default to `default_dataset_directory`.
    Returns:
        the file path of the dataset
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


if __name__ == '__main__':
    interval_choices = [Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_3MINUTE, Client.KLINE_INTERVAL_5MINUTE,
                        Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_30MINUTE, Client.KLINE_INTERVAL_1HOUR,
                        Client.KLINE_INTERVAL_2HOUR, Client.KLINE_INTERVAL_4HOUR, Client.KLINE_INTERVAL_6HOUR,
                        Client.KLINE_INTERVAL_8HOUR, Client.KLINE_INTERVAL_12HOUR, Client.KLINE_INTERVAL_1DAY,
                        Client.KLINE_INTERVAL_3DAY, Client.KLINE_INTERVAL_1WEEK, Client.KLINE_INTERVAL_1MONTH]

    parser = argparse.ArgumentParser(
        description=download_raw_dataset.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('symbol', help='the currency pair')
    parser.add_argument('interval', choices=interval_choices, help='duration of each candlestick')
    parser.add_argument('start_timestamp_millis', type=int, help='start unix timestamp in milliseconds')
    parser.add_argument('--end_timestamp_millis', type=int, help='end unix timestamp in milliseconds')
    parser.add_argument('--dataset-directory', default=default_dataset_directory,
                        help='destination of the downloaded dataset', dest='dataset_directory')

    print(download_raw_dataset(**vars(parser.parse_args())))
