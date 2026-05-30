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

    if filters.get("years"):
        placeholders = ",".join(["?"] * len(filters["years"]))
        conditions.append(f"year IN ({placeholders})")
        params.extend([int(y) for y in filters["years"]])

    if filters.get("months"):
        placeholders = ",".join(["?"] * len(filters["months"]))
        conditions.append(f"month IN ({placeholders})")
        params.extend([int(m) for m in filters["months"]])

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


def get_filter_options(active_filters: dict) -> dict:
    """
    Dado el estado actual de los filtros activos, retorna las opciones
    disponibles para cada filtro como los valores distintos que existen
    en client_signals satisfaciendo todos los filtros activos excepto
    el filtro propio de cada dimensión.

    Para calcular las opciones de la dimensión X, se aplican todos los
    filtros activos EXCEPTO el filtro de X, y se retornan los valores
    distintos de X que tienen al menos un registro.

    Retorna un dict con claves:
      'years', 'sources', 'companies', 'products', 'actions'
    Cada valor es una lista de dicts {"label": ..., "value": ...}
    ordenada por conteo descendente.
    """
    import sqlite3
    import pandas as pd

    if not os.path.exists(DB_PATH):
        return {k: [] for k in ['years', 'sources', 'companies', 'products', 'actions']}

    def _query_options(dimension_col: str, exclude_key, filters: dict,
                       min_count: int = 1) -> list:
        """
        Consulta los valores distintos de dimension_col aplicando todos
        los filtros EXCEPTO el/los filtro(s) de exclude_key (str o iterable).
        """
        exclude = {exclude_key} if isinstance(exclude_key, str) else set(exclude_key)
        restricted = {k: v for k, v in filters.items() if k not in exclude}
        base = f"""
            SELECT {dimension_col}, COUNT(*) as cnt
            FROM client_signals
            WHERE {dimension_col} IS NOT NULL
        """
        query_with_filters, params = _build_dynamic_query(base, restricted)
        query_final = (
            query_with_filters
            + f" GROUP BY {dimension_col}"
            + f" HAVING COUNT(*) >= {min_count}"
            + f" ORDER BY cnt DESC"
        )
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("PRAGMA busy_timeout = 5000")
                df = pd.read_sql_query(query_final, conn, params=params)
            return df[dimension_col].tolist(), df["cnt"].tolist()
        except Exception as e:
            print(f"[get_filter_options] Error en {dimension_col}: {e}")
            return [], []

    # Lista de acciones válidas del dominio — evita valores basura del dataset
    VALID_ACTIONS = {
        'formal_complaint', 'complaining', 'churning',
        'churning_due_to_price', 'churning_due_to_policy',
        'positive_review', 'negative_review', 'advocating',
        'seeking_help', 'venting', 'sharing_positive_experience',
        'searching_for_alternatives', 'reacting_to_price_change',
        'reacting_to_policy_change', 'neutral_review',
        'researching', 'discussing',
    }

    # Años disponibles (excluye el propio filtro de período en cualquiera de sus formas)
    year_vals, year_cnts = _query_options("year", ("period", "years", "months"), active_filters)
    year_opts = [
        {"label": str(y), "value": y}
        for y in sorted(year_vals)
        if y is not None
    ]

    # Fuentes disponibles (solo las con > 100 registros para no mostrar ruido)
    src_vals, src_cnts = _query_options("source", "sources", active_filters,
                                        min_count=100)
    src_opts = [
        {"label": f"{v} ({c:,})", "value": v}
        for v, c in zip(src_vals, src_cnts)
    ]

    # Empresas disponibles (solo las con > 500 registros, máx 30)
    co_vals, co_cnts = _query_options("company", "companies", active_filters,
                                      min_count=500)
    co_opts = [
        {
            "label": f"{v[:45]}{'…' if len(str(v)) > 45 else ''} ({c:,})",
            "value": v,
        }
        for v, c in zip(co_vals[:30], co_cnts[:30])
    ]

    # Productos disponibles (solo los con > 1000 registros, máx 20)
    pr_vals, pr_cnts = _query_options("product_service", "products",
                                      active_filters, min_count=1000)
    pr_opts = [
        {
            "label": f"{v[:50]}{'…' if len(str(v)) > 50 else ''} ({c:,})",
            "value": v,
        }
        for v, c in zip(pr_vals[:20], pr_cnts[:20])
    ]

    # Acciones disponibles (filtradas al dominio válido, mín 10 registros)
    ac_vals, ac_cnts = _query_options("customer_action", "actions",
                                      active_filters, min_count=10)
    ac_opts = [
        {"label": f"{v} ({c:,})", "value": v}
        for v, c in zip(ac_vals, ac_cnts)
        if v in VALID_ACTIONS
    ]

    return {
        "years":     year_opts,
        "sources":   src_opts,
        "companies": co_opts,
        "products":  pr_opts,
        "actions":   ac_opts,
    }
