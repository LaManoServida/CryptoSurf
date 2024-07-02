import numpy as np
import talib

from logger import logger


def add_class_up(df, forecast_horizon, trading_fee_percentage, forecast_gap=0):
    """
    Calculate and add to the dataset a boolean class "up", which is true if any point in the forecast horizon goes up
    with respect to the "close" value, taking buying and selling fees into account.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        forecast_horizon (int): Size of the forecast horizon used to compute the "up" class.
        trading_fee_percentage (float): Fee as a percentage of the asset purchased, used in calculations.
        forecast_gap (int, optional): Number of time steps between the latest "close" value and the forecast horizon.
            Defaults to 0.

    Returns: pandas.DataFrame: The input DataFrame with the added "up" column. The last few rows are removed as it is
    not possible to compute "up" values out of them.
    """

    logger.info("Calculating the class 'up'")

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

    # get rid of the last few rows of df, as it was not possible to compute "up" for them # TODO: rollback
    df = df.iloc[:-forecast_gap - forecast_horizon]

    # add the new column
    df = df.assign(up=up_list)

    return df


def add_sma(df, period):
    df[f'sma_{period}'] = talib.SMA(df['close'], period)


def add_vama(df, period):
    volume_price = df['close'] * df['volume']
    df[f'vama_{period}'] = volume_price.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()


def add_tema(df, period):
    df[f'tema_{period}'] = talib.TEMA(df['close'], period)


def add_ema(df, period):
    df[f'ema_{period}'] = talib.EMA(df['close'], period)


def add_dema(df, period):
    df[f'dema_{period}'] = talib.DEMA(df['close'], period)


def add_mom(df, period=10):
    df[f'mom_{period}'] = talib.MOM(df['close'], period)


def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df[f'macd_{fast_period}_{slow_period}_{signal_period}'] = \
        talib.MACD(df['close'], fast_period, slow_period, signal_period)[0]


def add_percent_b(df, period=5, stddev_upper=2, stddev_lower=2, ma_type=0):
    upper_band, _, lower_band = talib.BBANDS(df['close'], period, stddev_upper, stddev_lower, ma_type)
    df[f'percent_b_{period}_{stddev_upper}_{stddev_lower}_{ma_type}'] = \
        (df['close'] - lower_band) / (upper_band - lower_band) * 100


def chaikin_oscillator(df):
    df['chaikin_oscillator'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])


def add_roc(df, period=10):
    df[f'roc_{period}'] = talib.ROC(df['close'], period)


def add_so(df, fastk_period=5, slow_k_period=3, slow_k_ma_type=0, slow_d_period=3, slow_d_ma_type=0):
    df[f'so_{fastk_period}_{slow_k_period}_{slow_k_ma_type}_{slow_d_period}_{slow_d_ma_type}'] = \
        talib.STOCH(df['high'], df['low'], df['close'], fastk_period, slow_k_period, slow_k_ma_type, slow_d_period,
                    slow_d_ma_type)[0]


def add_trix(df, period=30):
    df[f'trix_{period}'] = talib.TRIX(df['close'], period)


def add_rsi(df, period=14):
    df[f'rsi_{period}'] = talib.RSI(df['close'], period)


def add_williams_percent_r(df, period=14):
    df[f'williams_percent_r_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], period)


def add_lagged_values(df, column_name, period):
    df[f'{column_name}_lagged_{period}'] = df[column_name].shift(period)

# def add_wma(df, period):
#     weights = np.arange(1, period + 1)
#     df['wma'] = df['close'].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
#
#
#
# def add_kama(df, n):
#     fastest_sc = 2 / (2 + 1)
#     slowest_sc = 2 / (30 + 1)
#
#     change = abs(df['close'] - df['close'].shift(n))
#     volatility = df['close'].diff().abs().rolling(window=n).sum()
#
#     er = change / volatility
#     sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
#
#     df['KAMA'] = df['close'].copy()
#     for i in range(n, len(df)):
#         df['KAMA'].iloc[i] = df['KAMA'].iloc[i - 1] + sc.iloc[i] * (df['close'].iloc[i] - df['KAMA'].iloc[i - 1])
