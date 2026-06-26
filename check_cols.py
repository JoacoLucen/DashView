from src.database_manager import _sql_to_polars

print("--- Column Counts ---")
q = "SELECT COUNT(source), COUNT(company), COUNT(product_service), COUNT(customer_action) FROM client_signals"
print(_sql_to_polars(q, []))

print("\n--- Distinct Companies (first 5) ---")
q = "SELECT DISTINCT company FROM client_signals LIMIT 5"
print(_sql_to_polars(q, []))

print("\n--- Distinct Products (first 5) ---")
q = "SELECT DISTINCT product_service FROM client_signals LIMIT 5"
print(_sql_to_polars(q, []))
