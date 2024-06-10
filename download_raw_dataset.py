import argparse
import os
from datetime import datetime

import pandas as pd
from binance import Client
from dateutil.relativedelta import relativedelta

from config import api_key, api_secret, default_dataset_directory


def _calculate_start_time(interval, number_candles):
    # split interval into coefficient and unit (for example: '30m' --> 30, 'm')
    interval_coefficient, interval_unit = int(interval[:-1]), interval[-1]

    # total units of time
    num_units = interval_coefficient * number_candles + 1  # +1 because the most recent one is never complete

    # calculate time span of data
    time_interval = {
        'm': relativedelta(minutes=num_units),
        'h': relativedelta(hours=num_units),
        'd': relativedelta(days=num_units),
        'w': relativedelta(weeks=num_units),
        'M': relativedelta(months=num_units)
    }[interval_unit]

    # convert it to timestamp
    start_time = (datetime.now() - time_interval).timestamp()
    start_time = round(start_time * 1000)

    return start_time


def download_raw_dataset(symbol, interval, number_candles, dataset_directory=default_dataset_directory):
    """Download the latest candlestick historical data and save it by default to `default_dataset_directory`.
    Returns:
        the file path of the dataset
    """

    # calculate start time
    start_time = _calculate_start_time(interval, number_candles + 1)

    # create client
    client = Client(api_key, api_secret)

    # download candlestick data (exclude the last one as it is not complete yet)
    candles = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_time)[:-1]

    # get columns of interest and set data type
    candles = pd.DataFrame(
        candles,
        columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
    ).drop('ignore', axis=1)

    # save it to csv
    os.makedirs(dataset_directory, exist_ok=True)
    file_name = f'candles({symbol},{number_candles + 1},{interval},{start_time}).csv'
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
    parser.add_argument('interval', default=Client.KLINE_INTERVAL_30MINUTE, choices=interval_choices,
                        help='duration of each candlestick')
    parser.add_argument('number_candles', default=1000, type=int, help='number of last candlesticks')
    parser.add_argument('--dataset-directory', default=default_dataset_directory,
                        help='destination of the downloaded dataset', dest='dataset_directory')

    print(download_raw_dataset(**vars(parser.parse_args())))
