import sqlite3
import pandas as pd

try:
    from .database_manager import _execute_query, DB_PATH
except ImportError:
    from src.database_manager import _execute_query, DB_PATH


# =========================================================================
# PESTAÑA 1: MARKETING
# =========================================================================

def get_nps_proxy(filters: dict) -> dict:
    """NPS Proxy basado en etiquetas de sentimiento para mayor consistencia analítica."""
    query = """
        SELECT
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as promoters,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as detractors,
            COUNT(*) as total
        FROM client_signals
    """
    df = _execute_query(query, filters)
    total = int(df["total"].iloc[0]) if not df.empty else 0
    promoters = int(df["promoters"].iloc[0] or 0) if not df.empty else 0
    detractors = int(df["detractors"].iloc[0] or 0) if not df.empty else 0
    passives = max(0, total - promoters - detractors)

    pct_p = round(promoters / total * 100, 1) if total else 0.0
    pct_d = round(detractors / total * 100, 1) if total else 0.0
    nps = round(pct_p - pct_d, 1)

    breakdown_df = pd.DataFrame({
        "Segmento": ["Promotores", "Pasivos", "Detractores"],
        "Cantidad": [promoters, passives, detractors],
    })
    return {
        "total_signals": total,
        "nps_score": nps,
        "pct_promoters": pct_p,
        "pct_detractors": pct_d,
        "breakdown_df": breakdown_df,
    }


def get_complaint_velocity(filters: dict) -> pd.DataFrame:
    """Volumen trimestral de quejas y deserciones — detecta tendencias de empeoramiento."""
    query = """
        SELECT
            year,
            CASE
                WHEN month BETWEEN 1 AND 3 THEN 'Q1'
                WHEN month BETWEEN 4 AND 6 THEN 'Q2'
                WHEN month BETWEEN 7 AND 9 THEN 'Q3'
                ELSE 'Q4'
            END as quarter,
            COUNT(*) as quejas
        FROM client_signals
        WHERE customer_action IN (
            'complaining','formal_complaint','churning',
            'churning_due_to_price','churning_due_to_policy','venting'
        )
        GROUP BY year, quarter
        ORDER BY year, quarter
    """
    df = _execute_query(query, filters)
    if not df.empty:
        df["periodo"] = df["year"].astype(str) + " " + df["quarter"]
    return df


def get_sentiment_by_channel(filters: dict) -> pd.DataFrame:
    query = """
        SELECT source, AVG(sentiment_score) as avg_sentiment
        FROM client_signals
        WHERE sentiment_score IS NOT NULL
        GROUP BY source
        ORDER BY avg_sentiment DESC
    """
    return _execute_query(query, filters)


def get_monthly_activity_peaks(filters: dict) -> pd.DataFrame:
    query = "SELECT month, COUNT(*) as volumen FROM client_signals GROUP BY month ORDER BY month ASC"
    df = _execute_query(query, filters)
    meses_map = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    if not df.empty:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["mes_label"] = df["month"].map(meses_map)
    return df


def get_source_impact(filters: dict) -> pd.DataFrame:
    query = """
        SELECT
            source,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as pct_positive,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as pct_negative
        FROM client_signals
        GROUP BY source
    """
    return _execute_query(query, filters)


# =========================================================================
# PESTAÑA 2: DIRECCIÓN GENERAL
# =========================================================================

def get_general_direction_kpis(filters: dict) -> dict:
    query = """
        SELECT customer_action, COUNT(*) as cantidad
        FROM client_signals
        WHERE customer_action IN ('churning', 'churning_due_to_price', 'churning_due_to_policy')
        GROUP BY customer_action
    """
    df = _execute_query(query, filters)
    total_churn = df["cantidad"].sum() if not df.empty else 0

    mapeo_causas = {
        "churning": "Insatisfacción General",
        "churning_due_to_price": "Por Precio",
        "churning_due_to_policy": "Por Política"
    }
    if not df.empty:
        df["causa_label"] = df["customer_action"].map(mapeo_causas).fillna(df["customer_action"])
        df["pct"] = (df["cantidad"] / total_churn * 100).round(1) if total_churn > 0 else 0
    else:
        df = pd.DataFrame(columns=["customer_action", "cantidad", "causa_label", "pct"])

    return {"total_churn": total_churn, "distribucion": df}


def get_regulatory_exposure(filters: dict) -> dict:
    """% de señales provenientes de canales regulatorios (CFPB) — riesgo de cumplimiento."""
    query = """
        SELECT
            SUM(CASE WHEN source = 'CFPB' THEN 1 ELSE 0 END) as regulatorias,
            COUNT(*) as total
        FROM client_signals
    """
    df = _execute_query(query, filters)
    if df.empty:
        return {"total": 0, "regulatorias": 0, "pct": 0.0}
    total = int(df["total"].iloc[0] or 0)
    regulatorias = int(df["regulatorias"].iloc[0] or 0)
    pct = round(regulatorias / total * 100, 1) if total else 0.0
    return {"total": total, "regulatorias": regulatorias, "pct": pct}


def get_prechurn_signals_trend(filters: dict) -> pd.DataFrame:
    """Señales de alerta temprana por año: búsqueda de alternativas y reacción a cambios."""
    query = """
        SELECT year, COUNT(*) as prechurn
        FROM client_signals
        WHERE customer_action IN (
            'searching_for_alternatives',
            'reacting_to_price_change',
            'reacting_to_policy_change'
        )
        GROUP BY year
        ORDER BY year ASC
    """
    return _execute_query(query, filters)


def get_competitive_benchmark(filters: dict) -> pd.DataFrame:
    query = """
        SELECT company, AVG(sentiment_score) as avg_sentiment
        FROM client_signals
        WHERE sentiment_score IS NOT NULL
        GROUP BY company
        HAVING COUNT(*) > 500
        ORDER BY avg_sentiment ASC
        LIMIT 15
    """
    return _execute_query(query, filters)


def get_company_product_heatmap(filters: dict) -> pd.DataFrame:
    query = """
        SELECT company, product_service, AVG(sentiment_score) as avg_sentiment
        FROM client_signals
        GROUP BY company, product_service
        ORDER BY COUNT(*) DESC
        LIMIT 150
    """
    df = _execute_query(query, filters)
    if not df.empty:
        top_products = (
            df.groupby("product_service")["avg_sentiment"]
            .count().sort_values(ascending=False).head(8).index
        )
        df = df[df["product_service"].isin(top_products)]
    return df


# =========================================================================
# PESTAÑA 3: RETENCIÓN Y FACTURACIÓN
# =========================================================================

def get_escalation_rate(filters: dict) -> float:
    query = """
        SELECT customer_action, COUNT(*) as cantidad
        FROM client_signals
        WHERE customer_action IN ('complaining', 'formal_complaint')
        GROUP BY customer_action
    """
    df = _execute_query(query, filters)
    if not df.empty:
        df_idx = df.set_index('customer_action').reindex(['complaining', 'formal_complaint'], fill_value=0)
        comp = df_idx.loc['complaining', 'cantidad']
        form = df_idx.loc['formal_complaint', 'cantidad']
        total = comp + form
        if total > 0:
            return round((float(form) / float(total)) * 100, 1)
    return 0.0


def get_average_behavior_cycle(filters: dict) -> float:
    """Calcula días promedio entre primer señal negativa y churn por empresa (proxy de cliente)."""
    query = """
        SELECT AVG(JULIANDAY(date_churn) - JULIANDAY(date_neg)) as avg_days
        FROM (
            SELECT
                company,
                MIN(CASE WHEN sentiment_label='negative' THEN date END) as date_neg,
                MAX(CASE WHEN customer_action LIKE 'churning%' THEN date END) as date_churn
            FROM client_signals
            GROUP BY company
        )
        WHERE date_churn IS NOT NULL AND date_neg IS NOT NULL AND date_churn > date_neg
    """
    df = _execute_query(query, filters)
    if not df.empty and pd.notna(df["avg_days"].iloc[0]):
        return round(float(df["avg_days"].iloc[0]), 1)
    return 0.0


def get_product_risk_radar(filters: dict) -> pd.DataFrame:
    query = """
        SELECT
            product_service as product,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as neg_ratio,
            SUM(CASE WHEN customer_action LIKE 'churning%' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as churn_ratio,
            COUNT(*) as vol
        FROM client_signals
        GROUP BY product_service
    """
    df = _execute_query(query, filters)
    if not df.empty:
        max_vol = df["vol"].max() or 1
        df["norm_vol"] = df["vol"] / max_vol
        df["score"] = (df["neg_ratio"] * 40) + (df["churn_ratio"] * 40) + (df["norm_vol"] * 20)
        df["score"] = df["score"].round(1).clip(0, 100)
        df = df.sort_values(by="score", ascending=False).head(10)
    else:
        df = pd.DataFrame(columns=["product", "score", "neg_ratio", "churn_ratio", "vol", "norm_vol"])
    return df[["product", "score"]]


def get_complaint_topics(filters: dict) -> pd.DataFrame:
    """Categorías de queja usando keywords financieras específicas del dataset."""
    query = "SELECT text FROM client_signals WHERE sentiment_label = 'negative' LIMIT 20000"
    df = _execute_query(query, filters)

    topics = {
        "Cobros / Cargos Incorrectos": 0,
        "Atención al Cliente": 0,
        "Reporte Crediticio": 0,
        "Gestión de Cuenta": 0,
        "Préstamos / Hipotecas": 0,
        "App / Acceso Digital": 0,
    }
    if not df.empty:
        texts = df["text"].astype(str).str.lower().fillna("")
        topics["Cobros / Cargos Incorrectos"] = int(texts.str.contains(
            r"cobro|cargo|tarifa|interes|recargo|billing|charge|fee|overcharg|interest|payment|factura|price"
        ).sum())
        topics["Atención al Cliente"] = int(texts.str.contains(
            r"atencion|soporte|ejecutivo|ayuda|espera|support|call|wait|agent|help|service|representative|phone|rude|unhelpful"
        ).sum())
        topics["Reporte Crediticio"] = int(texts.str.contains(
            r"credito|credit|report|score|bureau|equifax|experian|transunion|dispute|inaccurate"
        ).sum())
        topics["Gestión de Cuenta"] = int(texts.str.contains(
            r"cuenta|account|cierre|closed|freeze|block|fraud|identity|stolen|unauthorized"
        ).sum())
        topics["Préstamos / Hipotecas"] = int(texts.str.contains(
            r"prestamo|mortgage|loan|hipoteca|foreclosure|modification|payment plan|deuda|debt|collector"
        ).sum())
        topics["App / Acceso Digital"] = int(texts.str.contains(
            r"app|web|login|password|online|digital|portal|slow|crash|error|bug|glitch"
        ).sum())

    return pd.DataFrame(list(topics.items()), columns=["Topic", "Frecuencia"]).sort_values("Frecuencia", ascending=False)


def get_state_intensity_map(filters: dict) -> pd.DataFrame:
    query = """
        SELECT
            REPLACE(REPLACE(country, 'United States - ', ''), 'United States – ', '') as estado,
            COUNT(*) as quejas
        FROM client_signals
        WHERE customer_action IN ('complaining', 'formal_complaint')
        AND country IS NOT NULL
        GROUP BY estado
        ORDER BY quejas DESC
        LIMIT 20
    """
    return _execute_query(query, filters)


# =========================================================================
# PESTAÑA 4: EQUIPO DE PRODUCTO / APP
# =========================================================================

def get_device_usage_comparison(filters: dict) -> pd.DataFrame:
    has_rating = False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(client_signals)")
            columns = [info[1] for info in cursor.fetchall()]
            has_rating = 'rating' in columns
    except Exception:
        pass

    # LIKE flexible para asegurar que capta GooglePlay y AppStore sin importar espacios
    if has_rating:
        query = """
            SELECT source, AVG(rating) as avg_rating, AVG(sentiment_score) as avg_sentiment
            FROM client_signals
            WHERE LOWER(source) LIKE '%appstore%' OR LOWER(source) LIKE '%google%play%'
            GROUP BY source
        """
    else:
        query = """
            SELECT source, AVG(sentiment_score) as avg_sentiment
            FROM client_signals
            WHERE LOWER(source) LIKE '%appstore%' OR LOWER(source) LIKE '%google%play%'
            GROUP BY source
        """
    return _execute_query(query, filters)


def get_rating_distribution(filters: dict) -> pd.DataFrame:
    """Distribución de calificaciones 1–5 estrellas en canales móviles."""
    query = """
        SELECT
            CAST(ROUND(rating) AS INTEGER) as estrellas,
            COUNT(*) as cantidad
        FROM client_signals
        WHERE (LOWER(source) LIKE '%appstore%' OR LOWER(source) LIKE '%google%play%')
        AND rating IS NOT NULL AND rating > 0
        GROUP BY CAST(ROUND(rating) AS INTEGER)
        ORDER BY estrellas
    """
    df = _execute_query(query, filters)
    if not df.empty:
        all_stars = pd.DataFrame({"estrellas": [1, 2, 3, 4, 5]})
        df = all_stars.merge(df, on="estrellas", how="left").fillna(0)
        df["cantidad"] = df["cantidad"].astype(int)
        df["estrella_label"] = df["estrellas"].astype(str) + " ★"
    else:
        df = pd.DataFrame(columns=["estrellas", "cantidad", "estrella_label"])
    return df


def get_app_reviews_nlp(filters: dict) -> pd.DataFrame:
    query = """
        SELECT text FROM client_signals
        WHERE sentiment_label = 'negative'
        AND (LOWER(source) LIKE '%appstore%' OR LOWER(source) LIKE '%google%play%')
        LIMIT 20000
    """
    df = _execute_query(query, filters)

    issues = {"Fallas (Crash)": 0, "Lentitud (Slow)": 0, "Errores (Bugs)": 0, "Acceso (Login)": 0}
    if not df.empty:
        texts = df["text"].astype(str).str.lower().fillna("")
        issues["Fallas (Crash)"]   = int(texts.str.contains(r"crash|crashea|caída|caida|cierra|close|quit").sum())
        issues["Lentitud (Slow)"]  = int(texts.str.contains(r"lento|lentitud|demora|delay|slow|lag").sum())
        issues["Errores (Bugs)"]   = int(texts.str.contains(r"error|falla|bug|glitch|problem").sum())
        issues["Acceso (Login)"]   = int(texts.str.contains(r"login|entrar|password|clave|usuario|access").sum())

    return pd.DataFrame(list(issues.items()), columns=["Problema", "Frecuencia"]).sort_values("Frecuencia", ascending=False)


def get_yoy_volume_and_sentiment(filters: dict) -> pd.DataFrame:
    query = """
        SELECT
            year,
            COUNT(*) as volumen,
            AVG(sentiment_score) as avg_sentiment
        FROM client_signals
        WHERE LOWER(source) LIKE '%appstore%' OR LOWER(source) LIKE '%google%play%'
        GROUP BY year
        ORDER BY year ASC
    """
    return _execute_query(query, filters)
