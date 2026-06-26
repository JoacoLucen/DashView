from src.metrics import get_device_usage_comparison
from src.database_manager import _sql_to_polars

# Mock filters
filters = {"period": [2010, 2025]}
# sources is removed in render_product_team
device_filters = {k: v for k, v in filters.items() if k != "sources"}

df = get_device_usage_comparison(device_filters)
print("--- Device Usage Comparison Result ---")
print(df)

print("\n--- Raw Group By Source ---")
print(_sql_to_polars("SELECT source, COUNT(*) FROM client_signals GROUP BY source", []))
