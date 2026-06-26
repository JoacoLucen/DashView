from src.database_manager import _sql_to_polars

print("--- Data Coverage ---")
q = """
SELECT
    source,
    COUNT(*) as total,
    COUNT(sentiment_score) as with_sentiment,
    COUNT(rating) as with_rating,
    COUNT(company) as with_company,
    COUNT(product_service) as with_product
FROM client_signals
GROUP BY source
"""
print(_sql_to_polars(q, []))
