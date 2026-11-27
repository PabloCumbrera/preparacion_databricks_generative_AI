# Databricks notebook source
# MAGIC %md
# MAGIC ![arquitectura_cine_ai_ai_search.png](./arquitectura_cine_ai_ai_search.png "arquitectura_cine_ai_ai_search.png")

# COMMAND ----------

# MAGIC %md
# MAGIC # Mosaic AI Vector Search — El filtro que rompe el retrieval
# MAGIC
# MAGIC **Caso único:** «¿Puedo cancelar una entrada online 90 minutos antes del inicio?»
# MAGIC
# MAGIC
# MAGIC **Objetivo observable:** tres consultas a la misma pregunta: sin filtro, con filtro correcto y con un filtro erróneo. Veremos por qué una query puede funcionar técnicamente y recuperar evidencia incorrecta.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparación mínima
# MAGIC
# MAGIC Este notebook necesita el SDK oficial `databricks-ai-search`. Instálalo antes de cargar el setup compartido.
# MAGIC La instalación es una operación sobre el entorno Python, no sobre el índice.
# MAGIC
# MAGIC **Orden:** instalar → reiniciar Python → `%run` del setup → conectar al índice.
# MAGIC

# COMMAND ----------

#%pip install databricks-ai-search

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_CineAI_Common_Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Conectar al índice que ya existe
# MAGIC
# MAGIC No se aprovisiona ningún recurso: únicamente obtenemos el cliente del índice.

# COMMAND ----------

if globals().get("AI_SEARCH_TEXT_COLUMN") != "text":
    raise RuntimeError(
        "The Common Setup is outdated. Import "
        "00_CineAI_Common_Setup_v3_Compatible_Text.dbc "
        "and run the setup notebook again."
    )

search_index = get_ai_search_index()

print(f"Connected to: {AI_SEARCH_INDEX_NAME}")
print(f"Indexed text column: {AI_SEARCH_TEXT_COLUMN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Pregunta real: sin filtros
# MAGIC
# MAGIC Lanzamos una búsqueda semántica y observamos los **tres fragmentos más relevantes**. Este paso recupera evidencia; todavía no hay respuesta generada por un LLM.
# MAGIC **Nota:** el índice real utiliza `text` como columna de embeddings. El setup actualizado devuelve esa columna con el alias `content` en el DataFrame para conservar las prácticas posteriores.
# MAGIC

# COMMAND ----------

QUESTION = "¿Puedo cancelar una entrada comprada online 90 minutos antes de la sesión?"

baseline = search_ai_search_chunks(search_index, QUESTION, num_results=3)
display(baseline)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Mismo query, dos filtros de metadata
# MAGIC
# MAGIC - **Filtro correcto**: `CE-CX-POL-001` (manual oficial de políticas).
# MAGIC - **Filtro incorrecto**: `DOC-PRESS-001` (press kit cinematográfico).
# MAGIC
# MAGIC No cambiamos la pregunta, el modelo de embeddings ni el número de resultados: **únicamente la fuente permitida por el filtro**.

# COMMAND ----------

correct = search_ai_search_chunks(
    search_index,
    QUESTION,
    document_id=AI_SEARCH_POLICY_DOCUMENT_ID,
    num_results=3,
)

wrong = search_ai_search_chunks(
    search_index,
    QUESTION,
    document_id=AI_SEARCH_PRESS_DOCUMENT_ID,
    num_results=3,
)

print("CORRECT FILTER: official policies")
display(correct)
print("INCORRECT FILTER: press kit")
display(wrong)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Medir el fallo silencioso
# MAGIC
# MAGIC Contamos cuántos fragmentos proceden de la fuente **autoritativa**. No inferimos que un score elevado equivalga a verdad, y no asumimos que el primer resultado contenga necesariamente toda la política.
# MAGIC
# MAGIC Un filtro incorrecto puede devolver cero resultados o documentos irrelevantes, **sin error técnico**.

# COMMAND ----------

import pandas as pd

comparison = pd.DataFrame([
    {
        "scenario": "No filter",
        "returned_chunks": len(baseline),
        "official_policy_chunks": int(baseline["document_id"].eq(AI_SEARCH_POLICY_DOCUMENT_ID).sum()),
    },
    {
        "scenario": "Correct policy filter",
        "returned_chunks": len(correct),
        "official_policy_chunks": int(correct["document_id"].eq(AI_SEARCH_POLICY_DOCUMENT_ID).sum()),
    },
    {
        "scenario": "Incorrect press filter",
        "returned_chunks": len(wrong),
        "official_policy_chunks": int(wrong["document_id"].eq(AI_SEARCH_POLICY_DOCUMENT_ID).sum()),
    },
])

display(comparison)

assert not correct.empty, (
    "The official policy filter returned no results. Verify document_id values "
    "and confirm the existing index has completed synchronization."
)
assert comparison.loc[2, "official_policy_chunks"] == 0, (
    "The incorrect filter unexpectedly returned an official policy document."
)

print("Validated: the source filter changes what evidence can reach the RAG pipeline.")