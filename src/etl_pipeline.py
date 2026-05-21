import os
import sqlite3
import glob
import zipfile
import shutil
from abc import ABC, abstractmethod
import polars as pl

# =============================================================================
# 1. INTERFAZ ABSTRACTA PARA FUENTES DE DATOS
# =============================================================================
class DataSourceConnector(ABC):
    @abstractmethod
    def read_data(self, source_path: str) -> pl.DataFrame:
        pass

# =============================================================================
# 2. IMPLEMENTACIÓN DE CONECTORES
# =============================================================================

class CSVConnector(DataSourceConnector):
    def read_data(self, source_path: str) -> pl.DataFrame:
        return pl.read_csv(source_path, infer_schema_length=10000, ignore_errors=True)

class ParquetConnector(DataSourceConnector):
    def read_data(self, source_path: str) -> pl.DataFrame:
        return pl.read_parquet(source_path)

class JSONConnector(DataSourceConnector):
    def read_data(self, source_path: str) -> pl.DataFrame:
        try:
            return pl.read_json(source_path)
        except Exception:
            return pl.read_ndjson(source_path)

# =============================================================================
# 3. ORQUESTADOR ETL CON SOPORTE ZIP
# =============================================================================
class ETLPipeline:
    def __init__(self, target_db_path: str = None, staging_dir: str = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_db_path = target_db_path or os.path.join(current_dir, "..", "data", "dashview.db")
        self.staging_dir = staging_dir or os.path.join(current_dir, "..", "data", "staging")
        self.UPLOADED_ZIP_PATH = os.path.join(self.staging_dir, "latest_data.zip")
        self.EXTRACTED_DIR = os.path.join(self.staging_dir, "extracted")

        os.makedirs(os.path.dirname(self.target_db_path), exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        
        # Nuevo Esquema Requerido
        self.REQUIRED_COLUMNS = [
            "date", "year", "month", "customer_action", "sentiment_score", 
            "sentiment_label", "source", "company", "product_service", "text", "country"
        ]

    def cleanup_staging(self) -> None:
        """Limpia la carpeta de staging y extracciones previas."""
        if os.path.exists(self.EXTRACTED_DIR):
            shutil.rmtree(self.EXTRACTED_DIR)
        os.makedirs(self.EXTRACTED_DIR, exist_ok=True)
        
        for f in glob.glob(os.path.join(self.staging_dir, "latest_data*")):
            try:
                os.remove(f)
            except: pass

    def process_zip_file(self) -> None:
        """Descomprime el archivo ZIP y procesa el primer archivo válido encontrado."""
        if not zipfile.is_zipfile(self.UPLOADED_ZIP_PATH):
            raise ValueError("El archivo cargado no es un ZIP válido.")

        with zipfile.ZipFile(self.UPLOADED_ZIP_PATH, 'r') as zip_ref:
            zip_ofiles = zip_ref.namelist()
            # Filtrar archivos ocultos o carpetas de sistema (como __MACOSX)
            valid_files = [f for f in zip_ofiles if not f.startswith('__') and os.path.splitext(f)[1].lower() in ['.csv', '.parquet', '.jsonl', '.json']]
            
            if not valid_files:
                raise FileNotFoundError("No se encontró un archivo .csv, .parquet o .jsonl dentro del ZIP.")
            
            # Extraer el primer archivo válido
            target_file = valid_files[0]
            zip_ref.extract(target_file, self.EXTRACTED_DIR)
            extracted_path = os.path.join(self.EXTRACTED_DIR, target_file)
            
            self._route_to_connector(extracted_path)

    def _route_to_connector(self, file_path: str) -> None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            connector = CSVConnector()
        elif ext == '.parquet':
            connector = ParquetConnector()
        elif ext in ['.json', '.jsonl']:
            connector = JSONConnector()
        else:
            raise ValueError(f"Formato interno {ext} no soportado.")
        
        self.run(connector, file_path)

    def run(self, connector: DataSourceConnector, source_input) -> None:
        df_raw = connector.read_data(source_input)
        if df_raw.height == 0:
            raise ValueError("El archivo interno está vacío.")
        df_processed = self._transform_and_clean(df_raw)
        self._load_to_sqlite(df_processed)

    def _transform_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        # Normalizar columnas
        df.columns = [c.strip().lower() for c in df.columns]
        req_cols_lower = [c.lower() for c in self.REQUIRED_COLUMNS]
        
        missing_cols = [col for col in req_cols_lower if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Esquema inválido. Faltan: {missing_cols}")

        df_selected = df.select(req_cols_lower)
        df_selected.columns = self.REQUIRED_COLUMNS

        # Limpieza de tipos y nulos
        df_selected = df_selected.with_columns([
            pl.col("date").str.to_date(strict=False),
            pl.col("sentiment_score").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("month").cast(pl.Int32, strict=False),
            pl.col("text").fill_null(""),
            pl.col("country").fill_null("Unknown")
        ])

        # Asegurar que year/month vengan de la fecha si fallan en el cast
        df_selected = df_selected.with_columns([
            pl.when(pl.col("year").is_null()).then(pl.col("date").dt.year()).otherwise(pl.col("year")).alias("year"),
            pl.when(pl.col("month").is_null()).then(pl.col("date").dt.month()).otherwise(pl.col("month")).alias("month")
        ])

        return df_selected.filter(pl.col("date").is_not_null())

    def _load_to_sqlite(self, df: pl.DataFrame) -> None:
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_signals (
                    date TEXT, year INTEGER, month INTEGER, customer_action TEXT, 
                    sentiment_score REAL, sentiment_label TEXT, source TEXT, 
                    company TEXT, product_service TEXT, text TEXT, country TEXT
                );
            """)
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("DELETE FROM client_signals;") 
            cursor.executemany("""
                INSERT INTO client_signals (date, year, month, customer_action, sentiment_score, sentiment_label, source, company, product_service, text, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, df.iter_rows())
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_main ON client_signals (company, year, month);")
            conn.commit()
        finally:
            conn.close()
