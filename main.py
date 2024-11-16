from scripts.data_preprocessing import load_data, preprocess_data
from scripts.feature_engineering import engineering_features
from scripts.forecast import generate_forecast
from scripts.model_training import train_model


def main():
    train, test, stores, features = load_data()
    train, test = preprocess_data(train, test, stores, features)

    train = engineering_features(train)
    test = engineering_features(test)

    train_model(train)
    generate_forecast(test)


if __name__ == '__main__':
    main()
