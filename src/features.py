import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_discount_vs_original"] = 1 - (
        df["listing_price"] / df["original_price"]
    )

    df["views_per_day"] = df["views_7d"] / df["listing_age_days"].clip(lower=1)

    df["favorites_per_view"] = df["favorites_7d"] / df["views_7d"].clip(lower=1)

    df["messages_per_view"] = df["messages_7d"] / df["views_7d"].clip(lower=1)

    df["seller_trust_score"] = (
        0.45 * (df["seller_rating"] / 5)
        + 0.35 * df["seller_response_rate"]
        + 0.20 * (df["seller_total_sales"].clip(0, 100) / 100)
        - 0.25 * df["seller_return_rate"]
    )

    df["demand_supply_pressure"] = (
        df["demand_index"] / df["competition_level"].clip(lower=0.05)
    )

    df["is_overpriced"] = (df["price_ratio"] > 1.15).astype(int)
    df["is_underpriced"] = (df["price_ratio"] < 0.85).astype(int)

    return df


FEATURE_COLUMNS = [
    "category",
    "condition",
    "brand",
    "product_age_months",
    "listing_price",
    "market_median_price",
    "price_ratio",
    "seller_rating",
    "seller_response_rate",
    "seller_total_sales",
    "seller_return_rate",
    "listing_age_days",
    "seasonality_score",
    "demand_index",
    "competition_level",
    "views_24h",
    "views_7d",
    "favorites_7d",
    "messages_7d",
    "price_discount_vs_original",
    "views_per_day",
    "favorites_per_view",
    "messages_per_view",
    "seller_trust_score",
    "demand_supply_pressure",
    "is_overpriced",
    "is_underpriced",
]


CATEGORICAL_COLUMNS = [
    "category",
    "condition",
    "brand",
]