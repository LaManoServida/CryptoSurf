import os.path

import h5py
import numpy as np
import pandas as pd

from config import default_dataset_directory


def transform_into_sliding_windows(input_dataset_path, window_size, stride=1,
                                   output_dataset_directory=default_dataset_directory):
    """
    Transform a dataset with class "up" into sliding windows and save it to HDF5.

    Args:
        input_dataset_path (str): Path to the input dataset file.
        window_size (int): Size of the sliding windows to be created.
        stride (int, optional): Stride of the sliding windows. Defaults to 1.
        output_dataset_directory (str, optional): Destination directory for the resulting dataset.
            Defaults to a predefined directory.

    Returns:
        str: The file path of the resulting HDF5 dataset.
    """

    # read dataframe
    df = pd.read_csv(input_dataset_path)

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

    # save the new dataset
    filename_prefix = f'windows({window_size},{stride})_'
    output_filename = filename_prefix + os.path.splitext(os.path.basename(input_dataset_path))[0] + '.hdf5'
    output_file_path = os.path.join(output_dataset_directory, output_filename)
    with h5py.File(output_file_path, 'w') as f:
        x_dataset = f.create_dataset('x', data=np.array(x_windows_list))
        x_dataset.attrs['columns'] = df.columns.tolist()
        f.create_dataset('up', data=np.array(up_list))

    return output_file_path
