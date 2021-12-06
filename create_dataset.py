import argparse
from inspect import getmembers

import pandas as pd
from binance import Client

from keys import api_key, api_secret


def main(args):
    """Download candle history and save it to csv"""
    client = Client(api_key, api_secret)  # TODO: support key file
    candles = client.get_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)

    # get columns of interest and set type
    candles = pd.DataFrame([candle[1:5] for candle in candles], columns=['open', 'high', 'low', 'close'], dtype=float)

    # save it to csv
    candles.to_csv('data.csv', index=False)  # TODO: support custom csv path


if __name__ == '__main__':  # TODO: support custom start time
    def parse_limit(limit):
        limit = int(limit)
        if 0 < limit <= 1000:
            return limit
        raise ValueError


    parser = argparse.ArgumentParser(description='Generates the dataset of candelstick historical data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('symbol', help='the currency pair')
    parser.add_argument('-i', '--interval', default='30m',
                        choices=[v for k, v in getmembers(Client) if k.startswith('KLINE_INTERVAL_')],
                        help='duration of each candlestick')
    parser.add_argument('-l', '--limit', default=1000, type=parse_limit, help='number of candlesticks')

    main(parser.parse_args())
