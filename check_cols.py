import sqlite3
import pandas as pd

db_path = r'C:\Users\joaco\OneDrive\Desktop\DashView\data\dashview.db'
conn = sqlite3.connect(db_path)

print("--- Column Counts ---")
q = "SELECT COUNT(source), COUNT(company), COUNT(product_service), COUNT(customer_action) FROM client_signals"
print(pd.read_sql_query(q, conn))

print("\n--- Distinct Companies (first 5) ---")
q = "SELECT DISTINCT company FROM client_signals LIMIT 5"
print(pd.read_sql_query(q, conn))

print("\n--- Distinct Products (first 5) ---")
q = "SELECT DISTINCT product_service FROM client_signals LIMIT 5"
print(pd.read_sql_query(q, conn))

conn.close()
