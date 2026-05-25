import os
import sqlite3
import pandas as pd
from src.metrics import get_device_usage_comparison

# Mock filters
filters = {"period": [2010, 2025]}
# sources is removed in render_product_team
device_filters = {k: v for k, v in filters.items() if k != "sources"}

df = get_device_usage_comparison(device_filters)
print("--- Device Usage Comparison Result ---")
print(df)

db_path = r'C:\Users\joaco\OneDrive\Desktop\DashView\data\dashview.db'
conn = sqlite3.connect(db_path)
print("\n--- Raw Group By Source ---")
print(pd.read_sql_query("SELECT source, COUNT(*) FROM client_signals GROUP BY source", conn))
conn.close()
