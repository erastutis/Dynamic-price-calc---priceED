-- Sell-through by price competitiveness bucket

SELECT
    category,
    CASE
        WHEN price_ratio < 0.80 THEN 'underpriced'
        WHEN price_ratio BETWEEN 0.80 AND 1.05 THEN 'competitive'
        WHEN price_ratio BETWEEN 1.05 AND 1.30 THEN 'slightly_overpriced'
        ELSE 'overpriced'
    END AS price_bucket,
    COUNT(*) AS listings,
    AVG(sold_30d) AS sell_through_30d,
    AVG(days_to_sell) AS avg_days_to_sell,
    AVG(listing_price) AS avg_listing_price
FROM marketplace_listings
GROUP BY category, price_bucket
ORDER BY category, sell_through_30d DESC;