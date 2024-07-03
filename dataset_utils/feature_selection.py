import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def three_step_feature_selection(df, target_column='close', exclude_columns=None, max_correlation=0.95, max_vif=10):
    """
    Perform a three-step feature selection process using Boruta, correlation analysis, and Variance Inflation Factor
    (VIF).

    Args:
        df (pandas.DataFrame): The input DataFrame containing features and target variable.
        target_column (str): The name of the target variable column. Default is 'close'.
        exclude_columns (list): List of column names to exclude from feature selection. Default is None.
        max_correlation (float): Maximum allowed correlation between features. Default is 0.95.
        max_vif (float): Maximum allowed Variance Inflation Factor. Default is 10.

    Returns:
        list: A list of selected feature names after the three-step selection process.

    Steps:
        0. Standardize features
        1. Boruta feature selection
        2. Remove highly correlated features
        3. Filter out variables with high VIF
    """

    if exclude_columns is None:
        exclude_columns = []

    # Prepare the data
    x = df.drop(columns=exclude_columns + [target_column])
    y = df[target_column]

    # Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Step 1: Boruta feature selection
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    boruta_selector.fit(x_scaled, y)

    # Get selected feature names
    selected_features = x.columns[boruta_selector.support_].tolist()

    # Step 2: Remove highly correlated features
    correlation_matrix = x[selected_features].corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > max_correlation)]
    selected_features = [feat for feat in selected_features if feat not in to_drop]

    # Step 3: Filter out variables with high VIF
    x_vif = x[selected_features]
    vif_data = pd.DataFrame()
    vif_data['feature'] = x_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(x_vif.values, i) for i in range(len(x_vif.columns))]
    selected_features = vif_data[vif_data['VIF'] <= max_vif]['feature'].tolist()

    # Add excluded columns and target column to the selected features
    selected_features.extend(exclude_columns)
    if target_column not in selected_features:
        selected_features.append(target_column)

    return selected_features
