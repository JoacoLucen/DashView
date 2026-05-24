import os
import sqlite3
import pandas as pd
import functools

_current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_current_dir, "..", "data", "dashview.db")


def _build_dynamic_query(base_query: str, filters: dict) -> tuple:
    conditions = []
    params = []

    if filters.get("period"):
        conditions.append("year BETWEEN ? AND ?")
        params.extend([int(filters["period"][0]), int(filters["period"][1])])

    if filters.get("sources"):
        placeholders = ",".join(["?"] * len(filters["sources"]))
        conditions.append(f"source IN ({placeholders})")
        params.extend(filters["sources"])

    if filters.get("companies"):
        placeholders = ",".join(["?"] * len(filters["companies"]))
        conditions.append(f"company IN ({placeholders})")
        params.extend(filters["companies"])

    if filters.get("products"):
        placeholders = ",".join(["?"] * len(filters["products"]))
        conditions.append(f"product_service IN ({placeholders})")
        params.extend(filters["products"])

    if filters.get("actions"):
        placeholders = ",".join(["?"] * len(filters["actions"]))
        conditions.append(f"customer_action IN ({placeholders})")
        params.extend(filters["actions"])

    if filters.get("sentiment") and filters["sentiment"] != "ALL":
        conditions.append("sentiment_label = ?")
        params.append(filters["sentiment"].lower())

    if not conditions:
        return base_query, []

    where_clause = f"({' AND '.join(conditions)})"
    query_upper = base_query.upper()
    keywords = [" GROUP BY ", " ORDER BY ", " LIMIT "]
    insert_pos = len(base_query)

    for kw in keywords:
        pos = query_upper.find(kw)
        if pos != -1 and pos < insert_pos:
            insert_pos = pos

    prefix = base_query[:insert_pos]
    suffix = base_query[insert_pos:]

    if " WHERE " in prefix.upper():
        full_query = f"{prefix} AND {where_clause}{suffix}"
    else:
        full_query = f"{prefix} WHERE {where_clause}{suffix}"

    return full_query, params


@functools.lru_cache(maxsize=128)
def _execute_query_cached(base_query: str, filters_tuple: tuple) -> pd.DataFrame:
    filters = dict(filters_tuple) if filters_tuple else {}
    query, params = _build_dynamic_query(base_query, filters)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA busy_timeout = 5000")
            return pd.read_sql_query(query, conn, params=params)
    except sqlite3.Error as e:
        print(f"[DB Error] SQL: {query}")
        raise RuntimeError(f"Error en BD: {e}")


def _execute_query(base_query: str, filters: dict = None) -> pd.DataFrame:
    filters_tuple = ()
    if filters:
        filters_tuple = tuple(
            sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in filters.items())
        )
    return _execute_query_cached(base_query, filters_tuple).copy()


def clear_cache() -> None:
    _execute_query_cached.cache_clear()
