import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.express as px

from src.pricing_engine import simulate_price_grid


st.set_page_config(
    page_title="priceED",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f1117;
        color: #f5f5f5;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em;
        color: #f8fafc;
    }

    h2, h3 {
        color: #e5e7eb;
        letter-spacing: -0.02em;
    }

    p, label, span {
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #151923;
        border-right: 1px solid #2a2f3a;
    }

    [data-testid="stMetric"] {
        background-color: #171b26;
        border: 1px solid #2a2f3a;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.7rem;
    }

    .stDataFrame {
        border: 1px solid #2a2f3a;
        border-radius: 12px;
    }

    div[data-testid="stExpander"] {
        background-color: #171b26;
        border: 1px solid #2a2f3a;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    div[data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("priceED")

st.markdown(
    """
    <div style="font-size: 1.05rem; color: #9ca3af; margin-top: -12px; margin-bottom: 28px;">
        Marketplace pricing model for testing how listing price affects sell-through,
        expected revenue and time to sell.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Product")

    category = st.selectbox(
        "Category",
        ["electronics", "fashion", "home", "sports", "books", "beauty"],
    )

    condition = st.selectbox(
        "Condition",
        ["new", "like_new", "good", "fair"],
    )

    brand = st.selectbox(
        "Brand",
        ["premium", "mid_range", "budget", "unknown"],
    )

    market_median_price = st.number_input(
        "Market median price (€)",
        min_value=3.0,
        max_value=1000.0,
        value=150.0,
        step=5.0,
    )

    product_age_months = st.slider(
        "Product age months",
        min_value=0,
        max_value=80,
        value=12,
    )

    st.header("Seller")

    seller_rating = st.slider(
        "Seller rating",
        min_value=2.5,
        max_value=5.0,
        value=4.6,
        step=0.1,
    )

    seller_response_rate = st.slider(
        "Response rate",
        min_value=0.2,
        max_value=1.0,
        value=0.9,
        step=0.01,
    )

    seller_total_sales = st.slider(
        "Total past sales",
        min_value=0,
        max_value=300,
        value=55,
    )

    seller_return_rate = st.slider(
        "Return rate",
        min_value=0.0,
        max_value=0.4,
        value=0.04,
        step=0.01,
    )

    st.header("Demand")

    listing_age_days = st.slider(
        "Listing age days",
        min_value=1,
        max_value=60,
        value=7,
    )

    demand_index = st.slider(
        "Demand index",
        min_value=0.4,
        max_value=1.8,
        value=1.15,
        step=0.05,
    )

    competition_level = st.slider(
        "Competition level",
        min_value=0.05,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    seasonality_score = st.slider(
        "Seasonality score",
        min_value=0.6,
        max_value=1.5,
        value=1.05,
        step=0.05,
    )

    views_24h = st.slider(
        "Views in last 24h",
        min_value=0,
        max_value=100,
        value=14,
    )

    views_7d = st.slider(
        "Views in last 7 days",
        min_value=0,
        max_value=700,
        value=95,
    )

    favorites_7d = st.slider(
        "Favorites in last 7 days",
        min_value=0,
        max_value=100,
        value=11,
    )

    messages_7d = st.slider(
        "Messages in last 7 days",
        min_value=0,
        max_value=50,
        value=4,
    )


listing = {
    "category": category,
    "condition": condition,
    "brand": brand,
    "product_age_months": product_age_months,
    "original_price": market_median_price * 1.6,
    "listing_price": market_median_price,
    "market_median_price": market_median_price,
    "price_ratio": 1.0,
    "seller_rating": seller_rating,
    "seller_response_rate": seller_response_rate,
    "seller_total_sales": seller_total_sales,
    "seller_return_rate": seller_return_rate,
    "listing_age_days": listing_age_days,
    "seasonality_score": seasonality_score,
    "demand_index": demand_index,
    "competition_level": competition_level,
    "views_24h": views_24h,
    "views_7d": views_7d,
    "favorites_7d": favorites_7d,
    "messages_7d": messages_7d,
}


simulation, best = simulate_price_grid(listing)


st.subheader("Suggested listing price")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Suggested price", f"€{best['candidate_price']:.2f}")
col2.metric("30-day sale probability", f"{best['sale_probability_30d'] * 100:.1f}%")
col3.metric("Estimated days to sell", f"{best['expected_days_to_sell']:.1f}")
col4.metric("Pricing score", f"{best['revenue_risk_score']:.3f}")


st.subheader("Price simulation")

fig_score = px.line(
    simulation,
    x="candidate_price",
    y="revenue_risk_score",
    markers=True,
    title="Pricing score by candidate price",
)

fig_score.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font_color="#f5f5f5",
    xaxis_title="Candidate price (€)",
    yaxis_title="Pricing score",
)

st.plotly_chart(fig_score, use_container_width=True)


fig_prob = px.line(
    simulation,
    x="candidate_price",
    y="sale_probability_30d",
    markers=True,
    title="30-day sale probability by price",
)

fig_prob.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font_color="#f5f5f5",
    xaxis_title="Candidate price (€)",
    yaxis_title="Sale probability",
)

st.plotly_chart(fig_prob, use_container_width=True)


fig_revenue = px.line(
    simulation,
    x="candidate_price",
    y="expected_revenue",
    markers=True,
    title="Expected revenue by price",
)

fig_revenue.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font_color="#f5f5f5",
    xaxis_title="Candidate price (€)",
    yaxis_title="Expected revenue (€)",
)

st.plotly_chart(fig_revenue, use_container_width=True)


st.subheader("Simulation data")

st.dataframe(
    simulation,
    use_container_width=True,
    hide_index=True,
)


with st.expander("How the price is selected"):
    st.write(
        "The app tests candidate prices around the market median price. "
        "For each price it estimates sale probability, days to sell and expected revenue. "
        "The selected price is the one with the highest pricing score."
    )