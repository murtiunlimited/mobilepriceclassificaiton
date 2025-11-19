import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


def categorical_summary(df_train):
    df_cat = df_train[['price_range', 'n_cores', 'blue', 'dual_sim',
                       'five_g', 'four_g', 'touch_screen', 'wifi']].astype(str)

    unique_counts = df_cat.nunique()
    unique_values = df_cat.apply(lambda x: x.unique())

    summary = pd.DataFrame({
        "Unique Count": unique_counts,
        "Values": unique_values
    })

    return summary


def numerical_summary(df_train, categorical_cols):
    df_num = df_train.drop(categorical_cols, axis=1)
    return df_num.describe().T.round(2)


def missing_values(df):
    return df.isnull().sum()


def calculate_vif(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    inflation = pd.DataFrame()
    inflation["feature"] = X_train.columns
    inflation["VIF"] = [
        variance_inflation_factor(X_scaled, i)
        for i in range(X_scaled.shape[1])
    ]

    return inflation
