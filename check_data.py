import sqlite3
import pandas as pd
import os

db_path = r'C:\Users\joaco\OneDrive\Desktop\DashView\data\dashview.db'
conn = sqlite3.connect(db_path)

print("\n--- Ratings count by source ---")
q = """
SELECT source, COUNT(rating) as rating_count, AVG(rating) as avg_rating
FROM client_signals 
WHERE REPLACE(LOWER(source), ' ', '') IN ('appstore', 'googleplay') 
GROUP BY source
"""
print(pd.read_sql_query(q, conn))

conn.close()
