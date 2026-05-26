import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor

from src.features import create_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def train_days_model(
    data_path="data/synthetic/marketplace_listings.csv",
    model_path="models/days_to_sell_model.pkl",
):
    df = pd.read_csv(data_path)
    df = create_features(df)

    # Days-to-sell žinome tik tiems listingams, kurie buvo parduoti
    df = df[df["sold_30d"] == 1].copy()
    df = df.dropna(subset=["days_to_sell"])

    target = "days_to_sell"

    X = df[FEATURE_COLUMNS]
    y = df[target]

    numeric_columns = [
        col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                "passthrough",
                numeric_columns,
            ),
        ]
    )

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nDAYS-TO-SELL MODEL")
    print("------------------")
    print(f"Rows used: {len(df)}")
    print(f"Average days-to-sell: {y.mean():.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\nModel saved to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train_days_model()