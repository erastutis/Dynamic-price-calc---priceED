import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from src.features import create_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def train_sale_model(
    data_path="data/synthetic/marketplace_listings.csv",
    model_path="models/sale_probability_model.pkl",
):
    df = pd.read_csv(data_path)
    df = create_features(df)

    target = "sold_30d"

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

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        class_weight="balanced",
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
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\nSALE PROBABILITY MODEL")
    print("----------------------")
    print(f"Rows: {len(df)}")
    print(f"Target positive rate: {y.mean():.3f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\nModel saved to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train_sale_model()