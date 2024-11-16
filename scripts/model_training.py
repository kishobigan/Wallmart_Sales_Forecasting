from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import joblib


def train_model(train):
    x = train.drop(["Weekly_Sales", "Date"], axis=1)
    y = train["Weekly_Sales"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    try:
        rmse = root_mean_squared_error(y_val, y_pred)
    except ImportError:
        # Backward-compatible method
        rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE: {rmse}")

    joblib.dump(model, "results/model.pkl")
    return model
