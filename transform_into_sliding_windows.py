import numpy as np


def transform_into_sliding_windows(df, window_size, stride=1):
    """
    Transform a dataset with class "up" into sliding windows and return the results as numpy arrays.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset with an 'up' column.
        window_size (int): Size of the sliding windows to be created.
        stride (int, optional): Stride of the sliding windows. Defaults to 1.

    Returns:
        tuple: A 3-tuple containing:
            - numpy.ndarray: X windows data
            - numpy.ndarray: up values corresponding to each window
            - list: Column names of the input DataFrame (excluding 'up')
    """

    # separate the dataframe from the class "up"
    up_series = df['up']
    df = df.drop('up', axis=1)

    # get its numpy array representation
    df_array = df.values

    # initialize the lists
    x_windows_list = []
    up_list = []

    # iterate over the dataset
    for i in range(0, len(df) - (window_size - 1), stride):
        # append new X window
        x_window = df_array[i:i + window_size]
        x_windows_list.append(x_window)

        # append new class "up" value
        up_list.append(up_series[i + window_size - 1])

    return np.array(x_windows_list), np.array(up_list), df.columns.tolist()
