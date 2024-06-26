import os.path

import numpy as np
import pandas as pd

from config import default_dataset_directory


def calculate_class_up(input_dataset_path, forecast_window_size, trading_fee_percentage, window_gap=0,
                       output_dataset_directory=default_dataset_directory):
    """
    Calculate and add to the dataset a boolean class "up", which is true if any point in the forecast window goes up
    with respect to the "close" value, taking buying and selling fees into account.

    Args:
        input_dataset_path (str): Path to the input dataset file.
        forecast_window_size (int): Size of the forecast window used to compute the "up" class.
        trading_fee_percentage (float): Fee as a percentage of the asset purchased, used in calculations.
        window_gap (int, optional): Number of time steps between the latest "close" value and the forecast window.
            Defaults to 0.
        output_dataset_directory (str, optional): Destination directory for the resulting dataset.
            Defaults to a predefined directory.

    Returns:
        str: The file path of the resulting dataset.
    """

    # read dataframe
    df = pd.read_csv(input_dataset_path)

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

    # save the new dataset
    filename_prefix = f'class({window_gap}+{forecast_window_size},{trading_fee_percentage})_'
    output_filename = filename_prefix + str(os.path.basename(input_dataset_path))
    output_file_path = os.path.join(output_dataset_directory, output_filename)
    df.to_csv(output_file_path, index=False)

    return output_file_path
