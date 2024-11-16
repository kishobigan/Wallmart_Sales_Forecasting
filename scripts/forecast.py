import pandas as pd
import joblib


def generate_forecast(test):
    model = joblib.load('results/model.pkl')
    x_test = test.drop(['Date'], axis=1)

    test['Weekly_Sales_Predicted'] = model.predict(x_test)

    test[['Date', 'Store', 'Weekly_Sales_Predicted']].to_csv('forecast.csv', index=False)