import pandas as pd

def clean_data(df_train, df_test):
    # Fill missing
    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_test.mean())

    # Remove duplicates
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()

    return df_train, df_test
