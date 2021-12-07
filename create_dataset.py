import argparse
from datetime import datetime

import pandas as pd
from binance import Client
from dateutil.relativedelta import relativedelta

from keys import api_key, api_secret


def calculate_start_time(interval, number_candles):
    # split interval into coeficient and unit (for example: '30m' --> 30, 'm')
    interval_coef, interval_unit = int(interval[:-1]), interval[-1]

    # total units of time
    num_units = interval_coef * number_candles + 1  # +1 because the most recent one is never complete

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


def create_dataset(args):
    """Download candlestick history and save it to csv"""

    # calculate start time
    start_time = calculate_start_time(args.interval, args.number_candles)

    # create client
    client = Client(api_key, api_secret)  # TODO: support key file

    # download candlestick data (exclude the last one as is not complete yet)
    candles = client.get_historical_klines(symbol=args.symbol, interval=args.interval, start_str=start_time)[:-1]

    # get columns of interest and set data type
    candles = pd.DataFrame([candle[1:5] for candle in candles], columns=['open', 'high', 'low', 'close'], dtype=float)

    # save it to csv
    candles.to_csv('data.csv', index=False)  # TODO: support custom csv path


if __name__ == '__main__':
    def get_interval_choices():
        return [Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_3MINUTE, Client.KLINE_INTERVAL_5MINUTE,
                Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_30MINUTE, Client.KLINE_INTERVAL_1HOUR,
                Client.KLINE_INTERVAL_2HOUR, Client.KLINE_INTERVAL_4HOUR, Client.KLINE_INTERVAL_6HOUR,
                Client.KLINE_INTERVAL_8HOUR, Client.KLINE_INTERVAL_12HOUR, Client.KLINE_INTERVAL_1DAY,
                Client.KLINE_INTERVAL_3DAY, Client.KLINE_INTERVAL_1WEEK, Client.KLINE_INTERVAL_1MONTH]


    parser = argparse.ArgumentParser(description='Generates the dataset of latest candelstick historical data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('symbol', help='the currency pair')
    parser.add_argument('-i', '--interval', default=Client.KLINE_INTERVAL_30MINUTE, choices=get_interval_choices(),
                        help='duration of each candlestick')
    parser.add_argument('-n', default=1000, type=int, help='number of last candlesticks', dest='number_candles')

    create_dataset(parser.parse_args())
