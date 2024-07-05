from dataset_utils.data_preprocessing import apply_hampel_filter, apply_sg_filter, apply_standard_scaler, \
    apply_robust_scaler
from dataset_utils.dataset_generation import download_raw_dataset, split_train_val_test
from dataset_utils.feature_creation import *


def main():
    df_original = download_raw_dataset('BTCUSDT', '1d', 1364774400000, 1720009820000)

    for forecast_horizon in [1, 3, 5, 7]:
        df = df_original.copy()
        add_class_up(df, forecast_horizon, 0, forecast_gap=0)
        add_new_features(df)
        df.dropna(inplace=True)
        apply_hampel_filter(df, 'close', window_size=15, n_sigmas=3)
        apply_sg_filter(df, 'close', window_size=51, polynomial_degree=5, mode='nearest')
        apply_standard_scaler(df, exclude_columns=['up'])
        apply_robust_scaler(df, exclude_columns=['up'])
        df_train, df_val, df_test = split_train_val_test(df, 0.7, 0.15)


def add_new_features(df):
    # Add moving averages
    add_sma(df, period=5)
    add_sma(df, period=10)
    add_vama(df, period=9)
    add_tema(df, period=9)
    add_ema(df, period=9)
    add_dema(df, period=9)

    # Add momentum and oscillator indicators
    add_mom(df, period=10)
    add_macd(df, fast_period=12, slow_period=26, signal_period=9)
    add_percent_b(df, period=5, stddev_upper=2, stddev_lower=2, ma_type=0)
    chaikin_oscillator(df)
    add_roc(df, period=10)
    add_so(df, fastk_period=5, slow_k_period=3, slow_k_ma_type=0, slow_d_period=3, slow_d_ma_type=0)
    add_trix(df, period=30)
    add_rsi(df, period=14)
    add_williams_percent_r(df, period=14)

    # Add lagged values
    add_lagged_values(df, column_name='close', period=1)
    add_lagged_values(df, column_name='close', period=2)
    add_lagged_values(df, column_name='close', period=3)


if __name__ == '__main__':
    main()
