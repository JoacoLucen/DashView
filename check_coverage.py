import sqlite3
import pandas as pd

db_path = r'C:\Users\joaco\OneDrive\Desktop\DashView\data\dashview.db'
conn = sqlite3.connect(db_path)

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
print(pd.read_sql_query(q, conn))

conn.close()
