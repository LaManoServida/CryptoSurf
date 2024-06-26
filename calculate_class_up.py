import numpy as np


def calculate_class_up(df, forecast_window_size, trading_fee_percentage, window_gap=0):
    """
    Calculate and add to the dataset a boolean class "up", which is true if any point in the forecast window goes up
    with respect to the "close" value, taking buying and selling fees into account.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        forecast_window_size (int): Size of the forecast window used to compute the "up" class.
        trading_fee_percentage (float): Fee as a percentage of the asset purchased, used in calculations.
        window_gap (int, optional): Number of time steps between the latest "close" value and the forecast window.
            Defaults to 0.

    Returns:
        pandas.DataFrame: The input DataFrame with the added "up" column.
    """

    # get the index of 'close' column
    close_index = df.columns.tolist().index('close')

    # get its numpy array representation
    df_array = df.values

    # iterate over the dataset
    up_list = []
    for i in range(0, len(df) - (window_gap + forecast_window_size)):
        forecast_window = df_array[i + 1 + window_gap:i + 1 + window_gap + forecast_window_size, close_index]
        profit_threshold = df_array[i, close_index] / (1 - trading_fee_percentage / 100) ** 2
        up_list.append(np.any(forecast_window > profit_threshold))

    # pad with NaNs
    up_list += [np.nan] * (window_gap + forecast_window_size)

    # add the new column
    df['up'] = up_list

    return df
