from dataset_utils.dataset_generation import download_raw_dataset
from dataset_utils.feature_creation import *


def main():
    end_dates = [1469923200000, 1493510400000, 1514678400000, 1636156800000]

    for i, end_date in enumerate(end_dates):
        df = download_raw_dataset('BTCUSDT', '1d', 1364774400000, end_date)

        if i == 3:
            forecast_horizons = [1, 3, 5, 7]
        else:
            forecast_horizons = [1]

        for forecast_horizon in forecast_horizons:
            add_class_up(df, forecast_horizon, 0, forecast_gap=0)
            add_new_features(df)

            # X_windows, y, column_names = transform_into_sliding_windows(df, 48, 1)


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
