from src.database_manager import _sql_to_polars

print("--- GooglePlay years ---")
q = "SELECT year, COUNT(*) FROM client_signals WHERE source='GooglePlay' GROUP BY year"
print(_sql_to_polars(q, []))

print("\n--- AppStore years ---")
q = "SELECT year, COUNT(*) FROM client_signals WHERE source='AppStore' GROUP BY year"
print(_sql_to_polars(q, []))
