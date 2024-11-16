import pandas as pd


def engineering_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.day
    df['Day'] = df['Date'].dt.day

    df = pd.get_dummies(df, columns=['Type'], drop_first=True)

    return df