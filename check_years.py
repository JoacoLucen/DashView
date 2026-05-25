import sqlite3
import pandas as pd

db_path = r'C:\Users\joaco\OneDrive\Desktop\DashView\data\dashview.db'
conn = sqlite3.connect(db_path)

print("--- GooglePlay years ---")
q = "SELECT year, COUNT(*) FROM client_signals WHERE source='GooglePlay' GROUP BY year"
print(pd.read_sql_query(q, conn))

print("\n--- AppStore years ---")
q = "SELECT year, COUNT(*) FROM client_signals WHERE source='AppStore' GROUP BY year"
print(pd.read_sql_query(q, conn))

conn.close()
