import os
import sqlite3
import glob
import zipfile
import shutil
import datetime
import polars as pl

_current_dir = os.path.dirname(os.path.abspath(__file__))
TARGET_DB_PATH = os.path.join(_current_dir, "..", "data", "dashview.db")
STAGING_DIR = os.path.join(_current_dir, "..", "data", "staging")
UPLOADED_ZIP_PATH = os.path.join(STAGING_DIR, "latest_data.zip")
EXTRACTED_DIR = os.path.join(STAGING_DIR, "extracted")

REQUIRED_COLUMNS = [
    "date", "year", "month", "customer_action", "sentiment_score",
    "sentiment_label", "source", "company", "product_service", "text", "country", "rating"
]

os.makedirs(os.path.dirname(TARGET_DB_PATH), exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)


def read_csv(source_path: str) -> pl.DataFrame:
    return pl.read_csv(source_path, infer_schema_length=10000, ignore_errors=True)


def read_parquet(source_path: str) -> pl.DataFrame:
    return pl.read_parquet(source_path)


def read_json_file(source_path: str) -> pl.DataFrame:
    try:
        return pl.read_json(source_path)
    except Exception:
        return pl.read_ndjson(source_path)


def _route_to_reader(file_path: str) -> pl.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return read_csv(file_path)
    elif ext == '.parquet':
        return read_parquet(file_path)
    elif ext in ['.json', '.jsonl']:
        return read_json_file(file_path)
    raise ValueError(f"Formato interno {ext} no soportado.")


def cleanup_staging() -> None:
    if os.path.exists(EXTRACTED_DIR):
        shutil.rmtree(EXTRACTED_DIR)
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(STAGING_DIR, "latest_data*")):
        try:
            os.remove(f)
        except Exception:
            pass


_DTYPE_MAP = {
    "date": pl.Utf8, "year": pl.Int32, "month": pl.Int32,
    "customer_action": pl.Utf8, "sentiment_score": pl.Float64,
    "sentiment_label": pl.Utf8, "source": pl.Utf8, "company": pl.Utf8,
    "product_service": pl.Utf8, "text": pl.Utf8, "country": pl.Utf8,
    "rating": pl.Float64,
}


def _transform_and_clean(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({c: c.strip().lower() for c in df.columns})

    for col_name in REQUIRED_COLUMNS:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=_DTYPE_MAP[col_name]).alias(col_name))

    df = df.select(REQUIRED_COLUMNS)

    df = df.with_columns([
        pl.col("sentiment_score").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("year").cast(pl.Int32, strict=False),
        pl.col("month").cast(pl.Int32, strict=False),
        pl.col("rating").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("text").cast(pl.Utf8).fill_null(""),
        pl.col("country").cast(pl.Utf8).fill_null("Unknown"),
    ])

    df = df.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("_date_parsed")
    )

    df = df.with_columns([
        pl.when(pl.col("year").is_null())
            .then(pl.col("_date_parsed").dt.year())
            .otherwise(pl.col("year")).alias("year"),
        pl.when(pl.col("month").is_null())
            .then(pl.col("_date_parsed").dt.month())
            .otherwise(pl.col("month")).alias("month"),
    ])

    df = df.with_columns(
        pl.col("_date_parsed").cast(pl.Utf8).alias("date")
    ).drop("_date_parsed")

    return df.filter(pl.col("date").is_not_null())


# ── Dataset schema helpers ─────────────────────────────────────────────────────

def _ensure_datasets_schema(cursor: sqlite3.Cursor) -> None:
    """Create datasets metadata table and ensure client_signals has dataset_id column."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            loaded_at TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_signals (
            dataset_id INTEGER,
            date TEXT, year INTEGER, month INTEGER, customer_action TEXT,
            sentiment_score REAL, sentiment_label TEXT, source TEXT,
            company TEXT, product_service TEXT, text TEXT, country TEXT, rating REAL
        );
    """)
    # Migration for existing DBs that lack the dataset_id column
    try:
        cursor.execute("ALTER TABLE client_signals ADD COLUMN dataset_id INTEGER;")
    except Exception:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_main ON client_signals (company, year, month);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_dataset ON client_signals (dataset_id);")


# ── Public dataset management API ─────────────────────────────────────────────

def get_datasets() -> list:
    if not os.path.exists(TARGET_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(TARGET_DB_PATH)
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'"
        ).fetchone()
        if not has_table:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT id, name, loaded_at, row_count FROM datasets ORDER BY loaded_at DESC"
        ).fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "loaded_at": r[2], "row_count": r[3]} for r in rows]
    except Exception:
        return []


def delete_dataset(dataset_id: int) -> None:
    conn = sqlite3.connect(TARGET_DB_PATH)
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.execute("DELETE FROM client_signals WHERE dataset_id = ?;", (dataset_id,))
        conn.execute("DELETE FROM datasets WHERE id = ?;", (dataset_id,))
        conn.commit()
    finally:
        conn.close()


def delete_all_datasets() -> None:
    conn = sqlite3.connect(TARGET_DB_PATH)
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.execute("DELETE FROM client_signals;")
        conn.execute("DELETE FROM datasets;")
        conn.commit()
    finally:
        conn.close()


# ── Core load (accumulative) ───────────────────────────────────────────────────

def _load_to_sqlite(df: pl.DataFrame, dataset_name: str) -> None:
    conn = sqlite3.connect(TARGET_DB_PATH)
    cursor = conn.cursor()
    try:
        _ensure_datasets_schema(cursor)
        cursor.execute("BEGIN TRANSACTION;")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO datasets (name, loaded_at, row_count) VALUES (?, ?, ?);",
            (dataset_name, now, df.height),
        )
        dataset_id = cursor.lastrowid
        rows = [(dataset_id, *row) for row in df.iter_rows()]
        cursor.executemany("""
            INSERT INTO client_signals (dataset_id, date, year, month, customer_action, sentiment_score,
                sentiment_label, source, company, product_service, text, country, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, rows)
        conn.commit()
    finally:
        conn.close()


_VALID_EXTENSIONS = {'.csv', '.parquet', '.jsonl', '.json'}
_EXT_PRIORITY = {'.parquet': 0, '.csv': 1, '.json': 2, '.jsonl': 3}


def _pick_best_file(entries: list) -> str:
    candidates = [
        e for e in entries
        if not e.endswith('/')
        and not os.path.basename(e).startswith(('.', '__', '~'))
        and os.path.splitext(e)[1].lower() in _VALID_EXTENSIONS
    ]
    if not candidates:
        return None
    root_level = [c for c in candidates if '/' not in c.strip('/')]
    pool = root_level if root_level else candidates
    return min(pool, key=lambda f: _EXT_PRIORITY.get(os.path.splitext(f)[1].lower(), 99))


def _try_load_db_from_zip(zip_ref: zipfile.ZipFile, all_entries: list, dataset_name: str) -> bool:
    """Read client_signals from an embedded .db file and append it accumulatively."""
    db_entries = [
        e for e in all_entries
        if not e.endswith('/')
        and not os.path.basename(e).startswith(('.', '__', '~'))
        and os.path.splitext(e)[1].lower() == '.db'
    ]
    for db_entry in db_entries:
        zip_ref.extract(db_entry, EXTRACTED_DIR)
        src_path = os.path.join(EXTRACTED_DIR, db_entry)
        try:
            with sqlite3.connect(src_path) as src_conn:
                tables = [
                    r[0] for r in src_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                if 'client_signals' not in tables:
                    continue
                count = src_conn.execute("SELECT COUNT(*) FROM client_signals").fetchone()[0]
                if count == 0:
                    continue
                col_info = src_conn.execute("PRAGMA table_info(client_signals)").fetchall()
                existing_cols = {row[1] for row in col_info}
                readable_cols = [c for c in REQUIRED_COLUMNS if c in existing_cols]
                if not readable_cols:
                    continue
                col_str = ", ".join(readable_cols)
                rows = src_conn.execute(f"SELECT {col_str} FROM client_signals").fetchall()
                col_data = {col: [row[i] for row in rows] for i, col in enumerate(readable_cols)}
                df = pl.DataFrame(col_data)
                _load_to_sqlite(df, dataset_name)
                return True
        except Exception:
            continue
    return False


def process_zip_file(dataset_name: str = "Dataset") -> None:
    if not zipfile.is_zipfile(UPLOADED_ZIP_PATH):
        raise ValueError("El archivo cargado no es un ZIP válido.")

    with zipfile.ZipFile(UPLOADED_ZIP_PATH, 'r') as zip_ref:
        all_entries = zip_ref.namelist()

        if _try_load_db_from_zip(zip_ref, all_entries, dataset_name):
            return

        target_file = _pick_best_file(all_entries)
        if target_file is None:
            found = ', '.join(all_entries[:10]) or '(vacío)'
            raise FileNotFoundError(
                f"No se encontró un archivo de datos (.csv, .parquet, .json, .jsonl, .db) "
                f"dentro del ZIP. Archivos encontrados: {found}"
            )

        zip_ref.extract(target_file, EXTRACTED_DIR)
        extracted_path = os.path.join(EXTRACTED_DIR, target_file)

        df_raw = _route_to_reader(extracted_path)
        if df_raw.height == 0:
            raise ValueError(f"El archivo interno '{target_file}' está vacío.")

        df_processed = _transform_and_clean(df_raw)
        _load_to_sqlite(df_processed, dataset_name)
