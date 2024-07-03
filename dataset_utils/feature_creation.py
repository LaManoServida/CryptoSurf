import numpy as np
import talib

from logger import logger


def add_class_up(df, forecast_horizon, trading_fee_percentage, forecast_gap=0):
    """
    Calculate and add to the dataset the class "up", containing 0, 1 and NaN values. The values are 1 if any point in
    the forecast horizon goes up with respect to the "close" value, taking buying and selling fees into account,
    and 0 otherwise.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        forecast_horizon (int): Size of the forecast horizon used to compute the "up" class.
        trading_fee_percentage (float): Fee as a percentage of the asset purchased, used in calculations.
        forecast_gap (int, optional): Number of time steps between the latest "close" value and the forecast horizon.
            Defaults to 0.
    """

    logger.info('Calculating the class "up"')

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

    # add NaNs for the last few rows, as it was not possible to compute "up" for them
    up_list.extend([np.nan] * (forecast_gap + forecast_horizon))

    # add the new column in place
    df['up'] = up_list


def add_sma(df, period):
    """
    Calculate and add the Simple Moving Average (SMA) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int): The number of periods to use in the SMA calculation.
    """

    logger.info('Calculating Simple Moving Average (SMA)')

    df[f'sma_{period}'] = talib.SMA(df['close'], period)


def add_vama(df, period):
    """
    Calculate and add the Volume Adjusted Moving Average (VAMA) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price and volume data.
        period (int): The number of periods to use in the VAMA calculation.
    """

    logger.info('Calculating Volume Adjusted Moving Average (VAMA)')

    volume_price = df['close'] * df['volume']
    df[f'vama_{period}'] = volume_price.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()


def add_tema(df, period):
    """
    Calculate and add the Triple Exponential Moving Average (TEMA) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int): The number of periods to use in the TEMA calculation.
    """

    logger.info('Calculating Triple Exponential Moving Average (TEMA)')

    df[f'tema_{period}'] = talib.TEMA(df['close'], period)


def add_ema(df, period):
    """
    Calculate and add the Exponential Moving Average (EMA) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int): The number of periods to use in the EMA calculation.
    """

    logger.info('Calculating Exponential Moving Average (EMA)')

    df[f'ema_{period}'] = talib.EMA(df['close'], period)


def add_dema(df, period):
    """
    Calculate and add the Double Exponential Moving Average (DEMA) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int): The number of periods to use in the DEMA calculation.
    """

    logger.info('Calculating Double Exponential Moving Average (DEMA)')

    df[f'dema_{period}'] = talib.DEMA(df['close'], period)


def add_mom(df, period=10):
    """
    Calculate and add the Momentum indicator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods to use in the Momentum calculation. Defaults to 10.
    """

    logger.info('Calculating Momentum indicator')

    df[f'mom_{period}'] = talib.MOM(df['close'], period)


def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate and add the Moving Average Convergence/Divergence (MACD) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        fast_period (int, optional): The number of periods for the fast EMA. Defaults to 12.
        slow_period (int, optional): The number of periods for the slow EMA. Defaults to 26.
        signal_period (int, optional): The number of periods for the signal line. Defaults to 9.
    """

    logger.info('Calculating Moving Average Convergence/Divergence (MACD)')

    df[f'macd_{fast_period}_{slow_period}_{signal_period}'] = \
        talib.MACD(df['close'], fast_period, slow_period, signal_period)[0]


def add_percent_b(df, period=5, stddev_upper=2, stddev_lower=2, ma_type=0):
    """
    Calculate and add the %B indicator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods for the moving average. Defaults to 5.
        stddev_upper (float, optional): The number of standard deviations for the upper band. Defaults to 2.
        stddev_lower (float, optional): The number of standard deviations for the lower band. Defaults to 2.
        ma_type (int, optional): The moving average type. Defaults to 0 (Simple Moving Average).
    """

    logger.info('Calculating %B indicator')

    upper_band, _, lower_band = talib.BBANDS(df['close'], period, stddev_upper, stddev_lower, ma_type)
    df[f'percent_b_{period}_{stddev_upper}_{stddev_lower}_{ma_type}'] = \
        (df['close'] - lower_band) / (upper_band - lower_band) * 100


def chaikin_oscillator(df):
    """
    Calculate and add the Chaikin A/D Oscillator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price and volume data.
    """

    logger.info('Calculating Chaikin A/D Oscillator')

    df['chaikin_oscillator'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])


def add_roc(df, period=10):
    """
    Calculate and add the Rate of Change (ROC) indicator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods to use in the ROC calculation. Defaults to 10.
    """

    logger.info('Calculating Rate of Change (ROC) indicator')

    df[f'roc_{period}'] = talib.ROC(df['close'], period)


def add_so(df, fastk_period=5, slow_k_period=3, slow_k_ma_type=0, slow_d_period=3, slow_d_ma_type=0):
    """
    Calculate and add the Stochastic Oscillator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        fastk_period (int, optional): The time period for the fast %K. Defaults to 5.
        slow_k_period (int, optional): The time period for the slow %K. Defaults to 3.
        slow_k_ma_type (int, optional): The moving average type for the slow %K. Defaults to 0 (Simple Moving Average).
        slow_d_period (int, optional): The time period for the slow %D. Defaults to 3.
        slow_d_ma_type (int, optional): The moving average type for the slow %D. Defaults to 0 (Simple Moving Average).
    """

    logger.info('Calculating Stochastic Oscillator')

    df[f'so_{fastk_period}_{slow_k_period}_{slow_k_ma_type}_{slow_d_period}_{slow_d_ma_type}'] = \
        talib.STOCH(df['high'], df['low'], df['close'], fastk_period, slow_k_period, slow_k_ma_type, slow_d_period,
                    slow_d_ma_type)[0]


def add_trix(df, period=30):
    """
    Calculate and add the TRIX indicator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods to use in the TRIX calculation. Defaults to 30.
    """

    logger.info('Calculating TRIX indicator')

    df[f'trix_{period}'] = talib.TRIX(df['close'], period)


def add_rsi(df, period=14):
    """
    Calculate and add the Relative Strength Index (RSI) to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods to use in the RSI calculation. Defaults to 14.
    """

    logger.info('Calculating Relative Strength Index (RSI)')

    df[f'rsi_{period}'] = talib.RSI(df['close'], period)


def add_williams_percent_r(df, period=14):
    """
    Calculate and add Williams %R indicator to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing price data.
        period (int, optional): The number of periods to use in the Williams %R calculation. Defaults to 14.
    """

    logger.info('Calculating Williams %R indicator')

    df[f'williams_percent_r_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], period)


def add_lagged_values(df, column_name, period):
    """
    Add lagged values of a specified column to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to lag.
        period (int): The number of periods to lag the values.
    """

    logger.info(f'Adding lagged values for {column_name}')

    df[f'{column_name}_lagged_{period}'] = df[column_name].shift(period)
