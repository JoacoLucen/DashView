import os
import pandas as pd
import numpy as np
import re
import sqlite3
from collections import Counter

try:
    from .database_manager import DatabaseManager
except ImportError:
    from src.database_manager import DatabaseManager

class MetricsCalculator:
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Calculador de métricas analíticas corporativas. 
        """
        self.db = db_manager or DatabaseManager()
        self.STOPWORDS = {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "no", 
            "si", "en", "de", "que", "es", "con", "por", "para", "como", "su", "sus", 
            "al", "del", "lo", "se", "me", "mi", "te", "tu", "muy", "mas", "más", "pero",
            "este", "esta", "esto", "fue", "fui", "está", "estoy", "donde", "cuando"
        }

    # =========================================================================
    # TABLA 1: MARKETING
    # =========================================================================

    def get_marketing_kpis(self, filters: dict) -> dict:
        query_vol = "SELECT COUNT(*) as total FROM client_signals"
        df_vol = self.db._execute_query(query_vol, filters)
        total_signals = int(df_vol["total"].iloc[0]) if not df_vol.empty else 0

        active_actions = "('complaining', 'churning', 'churning_due_to_price', 'churning_due_to_policy', 'formal_complaint', 'positive_review', 'negative_review', 'advocating', 'seeking_help', 'venting', 'sharing_positive_experience', 'reacting_to_price_change', 'reacting_to_policy_change', 'searching_for_alternatives', 'discussing')"
        
        query_active = f"""
            SELECT 
                SUM(CASE WHEN customer_action IN {active_actions} THEN 1 ELSE 0 END) as activos,
                COUNT(*) as total
            FROM client_signals
        """
        df_active = self.db._execute_query(query_active, filters)
        pct_activos = 0.0
        activos_count = 0
        total_count = 0
        
        if not df_active.empty:
            activos_count = df_active["activos"].iloc[0] or 0
            total_count = df_active["total"].iloc[0] or 0
            if total_count > 0:
                pct_activos = (activos_count / total_count) * 100

        df_pie = pd.DataFrame({
            "Tipo": ["Activos", "Pasivos"],
            "Cantidad": [activos_count, total_count - activos_count]
        })

        return {
            "total_signals": total_signals,
            "pct_activos": round(pct_activos, 1),
            "pie_data": df_pie
        }

    def get_sentiment_by_channel(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT source, AVG(sentiment_score) as avg_sentiment
            FROM client_signals
            WHERE sentiment_score IS NOT NULL
            GROUP BY source
            ORDER BY avg_sentiment DESC
        """
        return self.db._execute_query(query, filters)

    def get_monthly_activity_peaks(self, filters: dict) -> pd.DataFrame:
        query = "SELECT month, COUNT(*) as volumen FROM client_signals GROUP BY month ORDER BY month ASC"
        df = self.db._execute_query(query, filters)
        meses_map = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 
                     7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
        if not df.empty:
            df["month"] = pd.to_numeric(df["month"], errors="coerce")
            df["mes_label"] = df["month"].map(meses_map)
        return df

    def get_source_impact(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT 
                source,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as pct_positive,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as pct_negative
            FROM client_signals
            GROUP BY source
        """
        return self.db._execute_query(query, filters)

    def get_user_segmentation_distribution(self, filters: dict) -> pd.DataFrame:
        query = "SELECT customer_action, COUNT(*) as volumen FROM client_signals GROUP BY customer_action ORDER BY volumen DESC"
        df = self.db._execute_query(query, filters)
        if not df.empty:
            total_volumen = df["volumen"].sum()
            df["pct"] = (df["volumen"] / total_volumen * 100).round(1)
        return df

    def get_signals_volume_over_time(self, filters: dict) -> pd.DataFrame:
        query = "SELECT year, COUNT(*) as volumen FROM client_signals GROUP BY year ORDER BY year ASC"
        return self.db._execute_query(query, filters)

    # =========================================================================
    # TABLA 2: DIRECCIÓN GENERAL
    # =========================================================================

    def get_general_direction_kpis(self, filters: dict) -> dict:
        query = """
            SELECT customer_action, COUNT(*) as cantidad
            FROM client_signals
            WHERE customer_action IN ('churning', 'churning_due_to_price', 'churning_due_to_policy')
            GROUP BY customer_action
        """
        df = self.db._execute_query(query, filters)
        total_churn = df["cantidad"].sum() if not df.empty else 0
        
        mapeo_causas = {
            "churning": "Insatisfacción General",
            "churning_due_to_price": "Por Precio",
            "churning_due_to_policy": "Por Política"
        }
        if not df.empty:
            df["causa_label"] = df["customer_action"].map(mapeo_causas).fillna(df["customer_action"])
            df["pct"] = (df["cantidad"] / total_churn * 100).round(1) if total_churn > 0 else 0
        
        return {
            "total_churn": total_churn,
            "distribucion": df
        }

    def get_competitive_benchmark(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT company, AVG(sentiment_score) as avg_sentiment
            FROM client_signals
            WHERE sentiment_score IS NOT NULL
            GROUP BY company
            HAVING COUNT(*) > 500
            ORDER BY avg_sentiment ASC
            LIMIT 15
        """
        return self.db._execute_query(query, filters)

    def get_company_product_heatmap(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT company, product_service, AVG(sentiment_score) as avg_sentiment 
            FROM client_signals 
            GROUP BY company, product_service
            ORDER BY COUNT(*) DESC
            LIMIT 150
        """
        df = self.db._execute_query(query, filters)
        if not df.empty:
            top_products = df.groupby("product_service")["avg_sentiment"].count().sort_values(ascending=False).head(8).index
            df = df[df["product_service"].isin(top_products)]
        return df

    # =========================================================================
    # TABLA 3: RETENCIÓN Y FACTURACIÓN
    # =========================================================================

    def get_escalation_rate(self, filters: dict) -> float:
        query = """
            SELECT customer_action, COUNT(*) as cantidad 
            FROM client_signals 
            WHERE customer_action IN ('complaining', 'formal_complaint') 
            GROUP BY customer_action
        """
        df = self.db._execute_query(query, filters)
        if not df.empty:
            df_idx = df.set_index('customer_action').reindex(['complaining', 'formal_complaint'], fill_value=0)
            comp = df_idx.loc['complaining', 'cantidad']
            form = df_idx.loc['formal_complaint', 'cantidad']
            total = comp + form
            if total > 0: 
                return round((float(form) / float(total)) * 100, 1)
        return 0.0

    def get_average_behavior_cycle(self, filters: dict) -> float:
        query = """
            SELECT AVG(JULIANDAY(date_churn) - JULIANDAY(date_neg)) as avg_days
            FROM (
                SELECT 
                    source,
                    MIN(CASE WHEN sentiment_label='negative' THEN date END) as date_neg,
                    MAX(CASE WHEN customer_action LIKE 'churning%' THEN date END) as date_churn
                FROM client_signals
                GROUP BY source
            )
            WHERE date_churn IS NOT NULL AND date_neg IS NOT NULL AND date_churn > date_neg
        """
        df = self.db._execute_query(query, filters)
        if not df.empty and pd.notna(df["avg_days"].iloc[0]):
            return round(float(df["avg_days"].iloc[0]), 1)
        return 0.0

    def get_product_risk_radar(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT 
                product_service as product,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as neg_ratio,
                SUM(CASE WHEN customer_action LIKE 'churning%' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as churn_ratio,
                COUNT(*) as vol
            FROM client_signals
            GROUP BY product_service
        """
        df = self.db._execute_query(query, filters)
        if not df.empty:
            max_vol = df["vol"].max() or 1
            df["norm_vol"] = df["vol"] / max_vol
            df["score"] = ((df["neg_ratio"] * 40) + (df["churn_ratio"] * 40) + (df["norm_vol"] * 20))
            df["score"] = df["score"].round(1).clip(0, 100)
            df = df.sort_values(by="score", ascending=False).head(10)
        return df[["product", "score"]]

    def get_complaint_topics(self, filters: dict) -> pd.DataFrame:
        query = "SELECT text FROM client_signals WHERE sentiment_label = 'negative' LIMIT 20000"
        df = self.db._execute_query(query, filters)
        
        topics = {"Facturación / Cobros": 0, "Atención / Soporte": 0, "App / Interfaz / UX": 0, "Producto / Calidad": 0}
        if not df.empty:
            texts = df["text"].astype(str).str.lower().fillna("")
            topics["Facturación / Cobros"] = int(texts.str.contains("cobro|tarifa|interes|plata|dinero|pago|factura|precio|price|billing|charge|card|money|bank").sum())
            topics["Atención / Soporte"] = int(texts.str.contains("atencion|soporte|ejecutivo|ayuda|telefono|support|call|espera|atención|agent|help|service").sum())
            topics["App / Interfaz / UX"] = int(texts.str.contains("interfaz|boton|pantalla|color|app|lento|crash|ux|ui|error|login|entrar|acceso|slow").sum())
            topics["Producto / Calidad"] = int(texts.str.contains("malo|pobre|calidad|producto|servicio|funciona|broken|bad|quality").sum())
            
        return pd.DataFrame(list(topics.items()), columns=["Topic", "Frecuencia"]).sort_values("Frecuencia", ascending=False)

    def get_state_intensity_map(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT 
                REPLACE(REPLACE(country, 'United States - ', ''), 'United States – ', '') as estado, 
                COUNT(*) as quejas 
            FROM client_signals 
            WHERE customer_action IN ('complaining', 'formal_complaint')
            AND country IS NOT NULL
            GROUP BY country 
            ORDER BY quejas DESC
            LIMIT 20
        """
        return self.db._execute_query(query, filters)

    # =========================================================================
    # PESTAÑA 4: EQUIPO DE PRODUCTO / APP
    # =========================================================================

    def get_device_usage_comparison(self, filters: dict) -> pd.DataFrame:
        # FIX: Verificamos si la columna existe de forma segura
        has_rating = False
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(client_signals)")
                columns = [info[1] for info in cursor.fetchall()]
                has_rating = 'rating' in columns
        except:
            pass

        if has_rating:
            query = """
                SELECT source, AVG(rating) as avg_rating, AVG(sentiment_score) as avg_sentiment
                FROM client_signals 
                WHERE source IN ('AppStore', 'GooglePlay') 
                GROUP BY source
            """
        else:
            query = """
                SELECT source, AVG(sentiment_score) as avg_sentiment
                FROM client_signals 
                WHERE source IN ('AppStore', 'GooglePlay') 
                GROUP BY source
            """
        return self.db._execute_query(query, filters)

    def get_app_reviews_nlp(self, filters: dict) -> pd.DataFrame:
        query = "SELECT text FROM client_signals WHERE sentiment_label = 'negative' AND source IN ('AppStore', 'GooglePlay') LIMIT 20000"
        df = self.db._execute_query(query, filters)
        
        issues = {"Fallas (Crash)": 0, "Lentitud (Slow)": 0, "Errores (Bugs)": 0, "Acceso (Login)": 0}
        if not df.empty:
            texts = df["text"].astype(str).str.lower().fillna("")
            issues["Fallas (Crash)"] = int(texts.str.contains("crash|crashea|caída|caida|cierra|close|quit").sum())
            issues["Lentitud (Slow)"] = int(texts.str.contains("lento|lentitud|demora|delay|slow|lag").sum())
            issues["Errores (Bugs)"] = int(texts.str.contains("error|falla|bug|glitch|problem").sum())
            issues["Acceso (Login)"] = int(texts.str.contains("login|entrar|password|clave|usuario|access").sum())
            
        return pd.DataFrame(list(issues.items()), columns=["Problema", "Frecuencia"]).sort_values("Frecuencia", ascending=False)

    def get_yoy_volume_and_sentiment(self, filters: dict) -> pd.DataFrame:
        query = """
            SELECT 
                year, 
                COUNT(*) as volumen, 
                AVG(sentiment_score) as avg_sentiment
            FROM client_signals 
            WHERE source IN ('AppStore', 'GooglePlay')
            GROUP BY year 
            ORDER BY year ASC
        """
        return self.db._execute_query(query, filters)
