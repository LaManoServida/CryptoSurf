import argparse
import os.path

import h5py
import numpy as np
import pandas as pd

from config import default_dataset_directory


def transform_into_sliding_windows(input_dataset_path, window_size, stride,
                                   output_dataset_directory=default_dataset_directory):
    """
    Transform a dataset with class "up" into sliding windows and save it to HDF5.
    Returns:
        The file path of the resulting dataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=transform_into_sliding_windows.__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dataset_path', help='path of the input dataset')
    parser.add_argument('window_size', default=100, type=int, help='size of the sliding windows')
    parser.add_argument('-s', '--stride', default=1, type=int, help='stride of the sliding windows')
    parser.add_argument('--output-dataset-directory', default=default_dataset_directory,
                        help='destination directory of the resulting dataset', dest='output_dataset_directory')

    print(transform_into_sliding_windows(**vars(parser.parse_args())))
