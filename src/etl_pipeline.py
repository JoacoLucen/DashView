import os
import sqlite3
import glob
import zipfile
import shutil
from abc import ABC, abstractmethod
import polars as pl

# 1. Clase Abstracta donde cualquier conector nuevo debe implementar obligatoriamente el método read_data.
class DataSourceConnector(ABC):
    @abstractmethod
    # Definicion del metodo read_Data donde recibe la ruta del archivo y tiene que devolver obligatoriamente un DataFrame de Polars.
    def read_data(self, source_path: str) -> pl.DataFrame: 
        pass

# 2. Implementación de conectores específicos para cada formato, todos heredan de DataSourceConnector 
# y cumplen con la interfaz definida. Esto permite agregar nuevos formatos en el futuro sin modificar la lógica del ETL, 
# simplemente creando una nueva clase que implemente read_data.

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

# 3. La clase ETLPipeline se encarga de asegurar que el archivo subido sea un ZIP válido, extraer el archivo interno, 
# identificar su formato y delegar la lectura al conector correspondiente. Luego, transforma y limpia los datos antes de cargarlos en la base de datos SQLite.
class ETLPipeline:
    def __init__(self, target_db_path: str = None, staging_dir: str = None):
        """Configura todo el entorno necesario para que el pipeline pueda trabajar cuando se crea la clase.
        Define las rutas donde se guardará la base de datos final (dashview.db) y la carpeta temporal de trabajo (staging).
        Crea automáticamente las carpetas en el disco si aún no existen (os.makedirs).
        Define una lista llamada REQUIRED_COLUMNS con los nombres exactos y obligatorios de las columnas que la base de datos necesita para funcionar."""
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_db_path = target_db_path or os.path.join(current_dir, "..", "data", "dashview.db")
        self.staging_dir = staging_dir or os.path.join(current_dir, "..", "data", "staging")
        self.UPLOADED_ZIP_PATH = os.path.join(self.staging_dir, "latest_data.zip")
        self.EXTRACTED_DIR = os.path.join(self.staging_dir, "extracted")

        os.makedirs(os.path.dirname(self.target_db_path), exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.EXTRACTED_DIR, exist_ok=True)
        
        # Esquema Refinado (Incluyendo rating como columna opcional soportada)
        self.REQUIRED_COLUMNS = [
            "date", "year", "month", "customer_action", "sentiment_score", 
            "sentiment_label", "source", "company", "product_service", "text", "country", "rating"
        ]

    def cleanup_staging(self) -> None:
        """Borra de forma segura todos los archivos de los procesos anteriores para asegurarse de que no se mezclen datos viejos con los nuevos.
        Elimina por completo la carpeta de archivos extraídos, la vuelve a crear vacía y borra cualquier archivo temporal cuyo nombre empiece con latest_data."""
        
        if os.path.exists(self.EXTRACTED_DIR):
            shutil.rmtree(self.EXTRACTED_DIR)
        os.makedirs(self.EXTRACTED_DIR, exist_ok=True)
        
        for f in glob.glob(os.path.join(self.staging_dir, "latest_data*")):
            try:
                os.remove(f)
            except: pass

    def process_zip_file(self) -> None:
        """"Es el punto de partida cuando el usuario sube un archivo. Valida el archivo comprimido y extrae su contenido.
        Verifica si el archivo es un .zip válido.
        Revisa la lista de archivos internos y descarta archivos ocultos del sistema (como los que empiezan con __).
        Busca el primer archivo que sea válido (.csv, .parquet, .json, .jsonl), lo extrae en la carpeta temporal y le pasa la ruta al siguiente método (_route_to_connector)."""
        
        if not zipfile.is_zipfile(self.UPLOADED_ZIP_PATH):
            raise ValueError("El archivo cargado no es un ZIP válido.")

        with zipfile.ZipFile(self.UPLOADED_ZIP_PATH, 'r') as zip_ref:
            zip_ofiles = zip_ref.namelist()
            valid_files = [f for f in zip_ofiles if not f.startswith('__') and os.path.splitext(f)[1].lower() in ['.csv', '.parquet', '.jsonl', '.json']]
            
            if not valid_files:
                raise FileNotFoundError("No se encontró un archivo .csv, .parquet o .jsonl dentro del ZIP.")
            
            target_file = valid_files[0]
            zip_ref.extract(target_file, self.EXTRACTED_DIR)
            extracted_path = os.path.join(self.EXTRACTED_DIR, target_file)
            
            self._route_to_connector(extracted_path)

    def _route_to_connector(self, file_path: str) -> None:
        """"Analiza la extensión del archivo extraído para decidir qué herramienta/conector debe usar para leerlo.
        Es un método privado (indicado por el guion bajo _). Si el archivo termina en .csv, instancia un CSVConnector(); 
        si es .parquet, usa ParquetConnector(), etc. Una vez seleccionado el conector, invoca al método principal run."""
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
        """Coordina el flujo real del ETL secuencialmente
        1.Extract: Llama al conector para leer el archivo y transformarlo en un DataFrame en bruto de Polars (df_raw).
        2.Valida que el archivo no esté vacío (si tiene 0 filas, lanza un error).
        3.Transform: Llama a la limpieza de datos (_transform_and_clean).
        4.Load: Envía los datos limpios a la base de datos (_load_to_sqlite)."""
        
        df_raw = connector.read_data(source_input)
        if df_raw.height == 0:
            raise ValueError("El archivo interno está vacío.")
        df_processed = self._transform_and_clean(df_raw)
        self._load_to_sqlite(df_processed)

    def _transform_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica reglas estrictas para garantizar la calidad de la información antes de guardarla.
        Homogeneiza: Pasa todos los nombres de las columnas a minúsculas y les quita espacios vacíos laterales.
        Tolerancia a fallos: Si falta alguna columna obligatoria en el archivo de origen, la crea vacía (llena de None) en lugar de romper el programa.
        Conversión de tipos (Casting): Transforma textos a fechas reales (to_date), y asegura que las métricas de negocio como sentiment_score
        y rating sean números decimales (Float64), reemplazando valores rotos o nulos por 0.0.
        Lógica Inteligente: Si las columnas de año (year) o mes (month) están vacías, las calcula automáticamente extrayendo el año y mes de la columna date.
        Al final, elimina cualquier fila que no tenga una fecha válida (is_not_null())."""
        
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Gestionar columnas faltantes de forma flexible
        for col in self.REQUIRED_COLUMNS:
            if col.lower() not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col.lower()))

        req_cols_lower = [c.lower() for c in self.REQUIRED_COLUMNS]
        df_selected = df.select(req_cols_lower)
        df_selected.columns = self.REQUIRED_COLUMNS

        # Limpieza de tipos y nulos
        df_selected = df_selected.with_columns([
            pl.col("date").str.to_date(strict=False),
            pl.col("sentiment_score").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("month").cast(pl.Int32, strict=False),
            pl.col("rating").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("text").fill_null(""),
            pl.col("country").fill_null("Unknown")
        ])

        df_selected = df_selected.with_columns([
            pl.when(pl.col("year").is_null()).then(pl.col("date").dt.year()).otherwise(pl.col("year")).alias("year"),
            pl.when(pl.col("month").is_null()).then(pl.col("date").dt.month()).otherwise(pl.col("month")).alias("month")
        ])

        return df_selected.filter(pl.col("date").is_not_null())

    def _load_to_sqlite(self, df: pl.DataFrame) -> None:
        """Realiza la Carga de los datos en la base de datos SQLite
        Crea la tabla client_signals con su estructura correcta si no existía.
        Usa BEGIN TRANSACTION; para procesar miles de filas en la memoria antes de escribir en el disco
        Aplica un reemplazo total: borra todo lo que había antes (DELETE FROM) e inserta lo nuevo usando executemany combinado con df.iter_rows().
        Indexación: Crea un índice compuesto (idx_signals_main) sobre las columnas company, year y month. Esto es fundamental porque hace que las búsquedas y filtros en tu Dashboard sean instantáneos.
        Usa un bloque finally para asegurar que la conexión a la base de datos siempre se cierre, evitando corrupciones o bloqueos del archivo."""
        
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_signals (
                    date TEXT, year INTEGER, month INTEGER, customer_action TEXT, 
                    sentiment_score REAL, sentiment_label TEXT, source TEXT, 
                    company TEXT, product_service TEXT, text TEXT, country TEXT, rating REAL
                );
            """)
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("DELETE FROM client_signals;") 
            cursor.executemany("""
                INSERT INTO client_signals (date, year, month, customer_action, sentiment_score, sentiment_label, source, company, product_service, text, country, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, df.iter_rows())
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_main ON client_signals (company, year, month);")
            conn.commit()
        finally:
            conn.close()
