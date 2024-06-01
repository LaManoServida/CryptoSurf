import argparse
import os.path

import h5py
import numpy as np
import pandas as pd

from config import default_dataset_directory


def transform_into_sliding_windows(raw_dataset_path, x_window_size, forecast_window_size, buying_fee_percentage,
                                   window_gap, stride, output_dataset_directory=default_dataset_directory):
    """
    Transform and save a raw dataset into sliding windows, calculating two boolean classes "up" and "down",
    which are true if any point in the forecast window goes up or down, respectively, taking fees into account,
    with respect to the last point of the "X" window. If no point exceeds the threshold, "y" is 0.
    Returns:
        The file path of the resulting transformed dataset
    """
    # read dataframe
    df = pd.read_csv(raw_dataset_path)

    # get the index of 'close' column
    close_index = df.columns.tolist().index('close')

    # get its numpy array representation
    df_array = df.values

    # initialize the lists
    x_windows_list = []
    up_list = []
    down_list = []

    # iterate over the dataset
    for i in range(0, len(df) - (x_window_size + window_gap + forecast_window_size), stride):
        # append new X window
        x_window = df_array[i:i + x_window_size]
        x_windows_list.append(x_window)

        # append new classes
        forecast_window = df_array[
                          i + x_window_size + window_gap:i + x_window_size + window_gap + forecast_window_size,
                          close_index]
        threshold = x_window[-1, close_index] / (1 - buying_fee_percentage / 100)
        up_list.append(np.any(forecast_window > threshold))
        down_list.append(np.any(forecast_window < threshold))

    # save the data
    output_filename = os.path.splitext(os.path.basename(raw_dataset_path))[0].replace('candles', 'windows')
    output_file_path = os.path.join(output_dataset_directory, output_filename + '.hdf5')
    with h5py.File(output_file_path, 'w') as f:
        x_dataset = f.create_dataset('x', data=np.array(x_windows_list))
        x_dataset.attrs['columns'] = df.columns.tolist()
        f.create_dataset('up', data=np.array(up_list))
        f.create_dataset('down', data=np.array(down_list))

    return output_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=transform_into_sliding_windows.__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw_dataset_path', help='path of the raw dataset')
    parser.add_argument('x_window_size', default=100, type=int, help='size of the "X" window')
    parser.add_argument('forecast_window_size', default=5, type=int,
                        help='size of the forecast window used to compute the classes')
    parser.add_argument('buying_fee_percentage', type=float,
                        help='fee as a percentage of the asset purchased')
    parser.add_argument('-g', '--window-gap', default=0, type=int, dest='window_gap',
                        help='number of time steps between "X" window and forecast window')
    parser.add_argument('-s', '--stride', default=1, type=int, help='stride of the sliding windows')
    parser.add_argument('--output-dataset-directory', default=default_dataset_directory,
                        help='destination of the downloaded dataset', dest='output_dataset_directory')

    print(transform_into_sliding_windows(**vars(parser.parse_args())))
