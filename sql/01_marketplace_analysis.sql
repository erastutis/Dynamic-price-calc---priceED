-- Category-level pricing and sell-through analysis

SELECT
    category,
    COUNT(*) AS total_listings,
    AVG(listing_price) AS avg_listing_price,
    AVG(market_median_price) AS avg_market_price,
    AVG(price_ratio) AS avg_price_ratio,
    AVG(sold_30d) AS sell_through_30d,
    AVG(days_to_sell) AS avg_days_to_sell
FROM marketplace_listings
GROUP BY category
ORDER BY sell_through_30d DESC;