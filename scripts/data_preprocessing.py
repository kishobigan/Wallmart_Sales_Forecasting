import pandas as pd


def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    stores = pd.read_csv('data/stores.csv')
    features = pd.read_csv('data/features.csv')
    return train, test, stores, features


def preprocess_data(train, test, stores, features):
    train = train.merge(stores, on='Store').merge(features, on=['Store', 'Date'])
    test = test.merge(stores, on='Store').merge(features, on=['Store', 'Date'])

    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    return train, test
