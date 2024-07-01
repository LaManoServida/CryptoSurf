import numpy as np


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
