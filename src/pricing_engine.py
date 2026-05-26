import joblib
import numpy as np
import pandas as pd

from src.features import create_features, FEATURE_COLUMNS


def simulate_price_grid(
    listing: dict,
    sale_model_path="models/sale_probability_model.pkl",
    days_model_path="models/days_to_sell_model.pkl",
    min_ratio=0.70,
    max_ratio=1.30,
    steps=31,
    risk_penalty=0.25,
):
    sale_model = joblib.load(sale_model_path)
    days_model = joblib.load(days_model_path)

    market_price = listing["market_median_price"]

    price_grid = np.linspace(
        market_price * min_ratio,
        market_price * max_ratio,
        steps,
    )

    rows = []

    for price in price_grid:
        item = listing.copy()

        item["listing_price"] = price
        item["price_ratio"] = price / market_price

        # Simulate how buyer engagement changes when price changes.
        # Higher prices reduce views, favorites and messages.
        # Lower prices increase them.
        new_price_ratio = price / market_price
        base_price_ratio = listing.get("price_ratio", 1.0)

        relative_price_change = new_price_ratio / base_price_ratio

        engagement_multiplier = np.clip(
            relative_price_change ** -1.4,
            0.45,
            1.45,
        )

        item["views_24h"] = listing["views_24h"] * engagement_multiplier
        item["views_7d"] = listing["views_7d"] * engagement_multiplier
        item["favorites_7d"] = listing["favorites_7d"] * engagement_multiplier
        item["messages_7d"] = listing["messages_7d"] * engagement_multiplier

        temp_df = pd.DataFrame([item])
        temp_df = create_features(temp_df)

        X = temp_df[FEATURE_COLUMNS]

        sale_probability = sale_model.predict_proba(X)[0, 1]

        expected_days = days_model.predict(X)[0]
        expected_days = max(1, expected_days)

        expected_revenue = price * sale_probability

        revenue_score = expected_revenue / price_grid.max()
        speed_score = 1 / expected_days
        risk_score = 1 - sale_probability

        revenue_risk_score = (
            0.60 * revenue_score
            + 0.30 * sale_probability
            + 0.10 * speed_score
            - risk_penalty * risk_score
        )

        rows.append({
            "candidate_price": round(price, 2),
            "price_ratio": round(price / market_price, 3),
            "sale_probability_30d": round(sale_probability, 4),
            "expected_days_to_sell": round(expected_days, 1),
            "expected_revenue": round(expected_revenue, 2),
            "revenue_risk_score": round(revenue_risk_score, 4),
        })

    simulation = pd.DataFrame(rows)

    best_row = simulation.sort_values(
        "revenue_risk_score",
        ascending=False,
    ).iloc[0]

    return simulation, best_row


if __name__ == "__main__":
    example_listing = {
        "category": "electronics",
        "condition": "like_new",
        "brand": "mid_range",
        "product_age_months": 12,
        "original_price": 250,
        "listing_price": 150,
        "market_median_price": 150,
        "price_ratio": 1.0,
        "seller_rating": 4.6,
        "seller_response_rate": 0.9,
        "seller_total_sales": 55,
        "seller_return_rate": 0.04,
        "listing_age_days": 7,
        "seasonality_score": 1.05,
        "demand_index": 1.15,
        "competition_level": 0.45,
        "views_24h": 14,
        "views_7d": 95,
        "favorites_7d": 11,
        "messages_7d": 4,
    }

    simulation, best = simulate_price_grid(example_listing)

    print("\nPRICE SIMULATION")
    print("----------------")
    print(simulation.head())

    print("\nBest recommendation:")
    print(best)

    print("\nFull simulation:")
    print(simulation)