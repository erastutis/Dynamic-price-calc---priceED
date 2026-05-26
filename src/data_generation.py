import numpy as np
import pandas as pd


def generate_marketplace_data(n_rows=50000, random_state=42):
    np.random.seed(random_state)

    categories = ["electronics", "fashion", "home", "sports", "books", "beauty"]
    conditions = ["new", "like_new", "good", "fair"]
    brands = ["premium", "mid_range", "budget", "unknown"]

    df = pd.DataFrame({
        "listing_id": np.arange(1, n_rows + 1),
        "seller_id": np.random.randint(1, 5000, n_rows),
        "category": np.random.choice(categories, n_rows),
        "condition": np.random.choice(conditions, n_rows, p=[0.2, 0.35, 0.35, 0.1]),
        "brand": np.random.choice(brands, n_rows, p=[0.15, 0.35, 0.35, 0.15]),
        "product_age_months": np.random.gamma(2.0, 8.0, n_rows).astype(int),
        "seller_rating": np.clip(np.random.normal(4.4, 0.45, n_rows), 2.5, 5.0),
        "seller_response_rate": np.clip(np.random.normal(0.82, 0.18, n_rows), 0.2, 1.0),
        "seller_total_sales": np.random.poisson(35, n_rows),
        "seller_return_rate": np.clip(np.random.beta(2, 25, n_rows), 0, 0.4),
        "listing_age_days": np.random.randint(1, 60, n_rows),
        "seasonality_score": np.clip(np.random.normal(1.0, 0.18, n_rows), 0.6, 1.5),
        "demand_index": np.clip(np.random.normal(1.0, 0.25, n_rows), 0.4, 1.8),
        "competition_level": np.clip(np.random.normal(0.55, 0.22, n_rows), 0.05, 1.0),
    })

    category_base_price = {
        "electronics": 180,
        "fashion": 45,
        "home": 70,
        "sports": 85,
        "books": 18,
        "beauty": 30,
    }

    condition_multiplier = {
        "new": 1.00,
        "like_new": 0.82,
        "good": 0.62,
        "fair": 0.42,
    }

    brand_multiplier = {
        "premium": 1.55,
        "mid_range": 1.00,
        "budget": 0.68,
        "unknown": 0.52,
    }

    df["market_median_price"] = df["category"].map(category_base_price)
    df["market_median_price"] *= df["condition"].map(condition_multiplier)
    df["market_median_price"] *= df["brand"].map(brand_multiplier)
    df["market_median_price"] *= np.random.normal(1.0, 0.15, n_rows)

    df["original_price"] = df["market_median_price"] * np.random.uniform(1.1, 2.0, n_rows)
    df["listing_price"] = df["market_median_price"] * np.random.normal(1.0, 0.22, n_rows)
    df["listing_price"] = np.clip(df["listing_price"], 3, None)

    df["price_ratio"] = df["listing_price"] / df["market_median_price"]

    price_attractiveness = np.clip(1.5 - df["price_ratio"], 0.2, 1.8)

    seller_trust = (
        0.45 * (df["seller_rating"] / 5)
        + 0.35 * df["seller_response_rate"]
        + 0.20 * np.clip(df["seller_total_sales"] / 100, 0, 1)
        - 0.20 * df["seller_return_rate"]
    )

    df["views_24h"] = np.random.poisson(
        np.clip(8 * df["demand_index"] * price_attractiveness * df["seasonality_score"], 1, 80)
    )

    df["views_7d"] = df["views_24h"] * np.random.randint(4, 10, n_rows)
    df["favorites_7d"] = np.random.poisson(np.clip(df["views_7d"] * 0.08 * price_attractiveness, 0.2, 30))
    df["messages_7d"] = np.random.poisson(np.clip(df["views_7d"] * 0.035 * seller_trust, 0.1, 20))

    logit = (
        1.8
        - 2.4 * df["price_ratio"]
        + 1.25 * df["demand_index"]
        + 0.9 * seller_trust
        + 0.04 * df["favorites_7d"]
        + 0.08 * df["messages_7d"]
        - 0.9 * df["competition_level"]
        - 0.012 * df["product_age_months"]
    )

    sale_probability_30d = 1 / (1 + np.exp(-logit))

    df["sold_30d"] = np.random.binomial(1, sale_probability_30d)

    df["sold_14d"] = np.where(
        df["sold_30d"] == 1,
        np.random.binomial(1, np.clip(sale_probability_30d * 0.72, 0, 1)),
        0,
    )

    df["sold_7d"] = np.where(
        df["sold_14d"] == 1,
        np.random.binomial(1, np.clip(sale_probability_30d * 0.52, 0, 1)),
        0,
    )

    base_days = (
        38
        + 32 * df["price_ratio"]
        - 16 * df["demand_index"]
        - 0.45 * df["favorites_7d"]
        - 0.75 * df["messages_7d"]
        + 12 * df["competition_level"]
        - 8 * seller_trust
    )

    df["days_to_sell"] = np.clip(base_days + np.random.normal(0, 5, n_rows), 1, 120).round()
    df.loc[df["sold_30d"] == 0, "days_to_sell"] = np.nan

    df["final_sale_price"] = np.where(
        df["sold_30d"] == 1,
        df["listing_price"] * np.random.uniform(0.9, 1.0, n_rows),
        np.nan,
    )

    return df


if __name__ == "__main__":
    df = generate_marketplace_data(n_rows=50000)
    df.to_csv("data/synthetic/marketplace_listings.csv", index=False)

    print("Synthetic marketplace data generated.")
    print("Shape:", df.shape)
    print(df.head())
    print(df[["sold_7d", "sold_14d", "sold_30d", "days_to_sell"]].describe())
    