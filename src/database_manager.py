import os
import sqlite3
import pandas as pd
import functools

class DatabaseManager:
    """Su objetivo principal es recibir las consultas analíticas del dashboard y aplicarles dinámicamente los filtros que el usuario selecciona en la interfaz
    (como rango de años, empresas o sentimientos), asegurando que las respuestas sean instantáneas gracias a una estrategia inteligente de memoria caché."""
    
    def __init__(self, db_path: str = None):
        """
        Inicializa la conexión a la base de datos SQLite.
        Establece la ruta del archivo de base de datos relativa a este script.
        """
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, "..", "data", "dashview.db")
        else:
            self.db_path = db_path

    def _build_dynamic_query(self, base_query: str, filters: dict) -> tuple:
        """
        Construye de forma dinámica las cláusulas WHERE insertándolas correctamente
        antes de palabras clave como GROUP BY o ORDER BY para asegurar sintaxis SQL válida.
        """
        conditions = []
        params = []

        # 1. Filtro de Período (Años)
        if filters.get("period"):
            conditions.append("year BETWEEN ? AND ?")
            params.extend([int(filters["period"][0]), int(filters["period"][1])])

        # 2. Filtro de Source (Plataforma/Fuente)
        if filters.get("sources"):
            placeholders = ",".join(["?"] * len(filters["sources"]))
            conditions.append(f"source IN ({placeholders})")
            params.extend(filters["sources"])

        # 3. Filtro de Empresa
        if filters.get("companies"):
            placeholders = ",".join(["?"] * len(filters["companies"]))
            conditions.append(f"company IN ({placeholders})")
            params.extend(filters["companies"])

        # 4. Filtro de Producto / Servicio
        if filters.get("products"):
            placeholders = ",".join(["?"] * len(filters["products"]))
            conditions.append(f"product_service IN ({placeholders})")
            params.extend(filters["products"])

        # 5. Filtro de Acción del Cliente
        if filters.get("actions"):
            placeholders = ",".join(["?"] * len(filters["actions"]))
            conditions.append(f"customer_action IN ({placeholders})")
            params.extend(filters["actions"])

        # 6. Filtro de Sentimiento
        if filters.get("sentiment") and filters["sentiment"] != "ALL":
            conditions.append("sentiment_label = ?")
            params.append(filters["sentiment"].lower())

        # Si no hay condiciones, retornamos la consulta base sin modificaciones
        if not conditions:
            return base_query, []

        where_clause = f"({' AND '.join(conditions)})"
        
        # Analizamos la consulta base para insertar el WHERE en el lugar correcto
        query_upper = base_query.upper()
        keywords = [" GROUP BY ", " ORDER BY ", " LIMIT "]
        insert_pos = len(base_query)
        
        for kw in keywords:
            pos = query_upper.find(kw)
            if pos != -1 and pos < insert_pos:
                insert_pos = pos
        
        prefix = base_query[:insert_pos]
        suffix = base_query[insert_pos:]
        
        # Determinar si ya existe un WHERE en la consulta original
        if " WHERE " in prefix.upper():
            full_query = f"{prefix} AND {where_clause}{suffix}"
        else:
            full_query = f"{prefix} WHERE {where_clause}{suffix}"

        return full_query, params

    @functools.lru_cache(maxsize=128)
    def _execute_query_cached(self, base_query: str, filters_tuple: tuple) -> pd.DataFrame:
        """
        Versión cacheada de la ejecución de la consulta.
        Utiliza lru_cache para guardar en memoria los últimos 128 resultados y
        evitar tiempos de carga al cambiar de pestaña con los mismos filtros.
        """
        # Reconstruir el diccionario de filtros a partir de la tupla hasheable
        filters = dict(filters_tuple) if filters_tuple else {}
        query, params = self._build_dynamic_query(base_query, filters)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # El pragma ayuda a evitar bloqueos (database is locked) en accesos concurrentes
                conn.execute("PRAGMA busy_timeout = 5000")
                return pd.read_sql_query(query, conn, params=params)
        except sqlite3.Error as e:
            print(f"[DB Error] SQL: {query}")
            raise RuntimeError(f"Error en BD: {e}")

    def _execute_query(self, base_query: str, filters: dict = None) -> pd.DataFrame:
        """
        Punto de entrada para ejecutar consultas. Convierte el diccionario de filtros
        en una tupla hasheable para poder utilizar el mecanismo de caché nativo.
        """
        filters_tuple = ()
        if filters:
            # Ordenamos las llaves y convertimos listas en tuplas para crear un objeto inmutable
            filters_tuple = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in filters.items()))
            
        return self._execute_query_cached(base_query, filters_tuple)

    def clear_cache(self) -> None:
        """
        Limpia la memoria caché. Se debe llamar cuando se carga un nuevo dataset
        para asegurar que no se muestren datos antiguos.
        """
        self._execute_query_cached.cache_clear()
