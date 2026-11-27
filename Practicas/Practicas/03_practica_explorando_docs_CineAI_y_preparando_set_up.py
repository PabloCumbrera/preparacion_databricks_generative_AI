# Databricks notebook source
# MAGIC %md
# MAGIC ![arquitectura_cine_ai_preparacion.png](./arquitectura_cine_ai_preparacion.png "arquitectura_cine_ai_preparacion.png")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Crear el catálogo donde vamos a alojar y gobernar todo el proyecto
# MAGIC CREATE CATALOG IF NOT EXISTS cineai;
# MAGIC
# MAGIC -- Crear el esquema para los datos estructurados originales
# MAGIC CREATE SCHEMA IF NOT EXISTS cineai.raw;
# MAGIC
# MAGIC -- Crear el esquema para los documentos procesados y el RAG
# MAGIC CREATE SCHEMA IF NOT EXISTS cineai.rag;
# MAGIC
# MAGIC -- Crear el esquema para los agentes y aplicaciones de IA
# MAGIC CREATE SCHEMA IF NOT EXISTS cineai.app;
# MAGIC
# MAGIC -- Crear el esquema para los datos de evaluación
# MAGIC CREATE SCHEMA IF NOT EXISTS cineai.eval;
# MAGIC
# MAGIC -- Crear el volumen para almacenar documentos PDF, Word e imágenes
# MAGIC CREATE VOLUME IF NOT EXISTS cineai.raw.documents;

# COMMAND ----------


import re
import shutil
from pathlib import Path

# Ruta de los archivos originales en el Workspace
workspace_path = Path(
    "/Workspace/Users/pablo_cumbrera_databricks@outlook.com/"
    "preparacion_databricks_generative_AI/"
    "CineAI_data_and_documents"
)

# Ruta del Volume de destino
volume_path = Path("/Volumes/cineai/raw/documents")

# Crear el catálogo, los esquemas y el Volume
spark.sql("CREATE CATALOG IF NOT EXISTS cineai")

for schema in ["raw", "rag", "app", "eval"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS cineai.{schema}")

spark.sql("CREATE VOLUME IF NOT EXISTS cineai.raw.documents")

# Definir los archivos que vamos a cargar
source_files = {
    "data": [
        "movies.csv",
        "screenings.csv",
        "ticket_sales.csv",
        "reviews.csv",
        "evaluation_set.csv"
    ],
    "documents": [
        "cinema_policies.pdf",
        "movie_press_kit.docx",
        "eclipse_protocol_poster.png",
        "scanned_customer_claim.png"
    ]
}

# Copiar los archivos originales al Volume
for folder, filenames in source_files.items():
    source_folder = workspace_path / folder
    destination_folder = volume_path / folder

    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"Source folder not found: {source_folder}"
        )

    destination_folder.mkdir(parents=True, exist_ok=True)
    found_files = set()

    for source_file in source_folder.iterdir():
        if not source_file.is_file():
            continue

        normalized_stem = re.sub(
            r"\(\d+\)$", "", source_file.stem
        ).strip()
        normalized_name = normalized_stem + source_file.suffix.lower()

        if normalized_name not in filenames:
            continue

        if normalized_name in found_files:
            raise ValueError(
                f"Duplicate source file: {normalized_name}"
            )

        destination_file = destination_folder / normalized_name
        shutil.copyfile(source_file, destination_file)

        found_files.add(normalized_name)
        print(f"Copied: {normalized_name}")

    missing_files = set(filenames) - found_files

    if missing_files:
        raise FileNotFoundError(
            f"Missing files in {source_folder}: {sorted(missing_files)}"
        )

print("Catalog, schemas, volume, and files are ready.")


# COMMAND ----------


# Define source files, target tables, and expected records
tables = {
    "movies": ("cineai.raw.movies", 36),
    "screenings": ("cineai.raw.screenings", 490),
    "ticket_sales": ("cineai.raw.ticket_sales", 3000),
    "reviews": ("cineai.raw.reviews", 720),
    "evaluation_set": ("cineai.eval.evaluation_set", 60)
}

# Read CSV files and create managed Delta tables
for filename, (table_name, expected_count) in tables.items():
    source_path = f"/Volumes/cineai/raw/documents/data/{filename}.csv"

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .option("mode", "FAILFAST")
        .csv(source_path)
    )

    # Validate source records before writing
    source_count = df.count()

    if source_count != expected_count:
        raise ValueError(
            f"{table_name}: expected {expected_count} records, "
            f"found {source_count}"
        )

    # Create or overwrite the Delta table
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

    # Validate the final table
    actual_count = spark.table(table_name).count()

    if actual_count != expected_count:
        raise ValueError(
            f"{table_name}: expected {expected_count} records, "
            f"found {actual_count}"
        )

    print(f"Created: {table_name} ({actual_count} records)")

print("All CineAI Delta tables are ready.")
