# Databricks notebook source
# MAGIC %md
# MAGIC # CineAI · Configuración común
# MAGIC
# MAGIC Funciones compartidas de inferencia y retrieval para las prácticas. No crea endpoints, tablas ni recursos persistentes.

# COMMAND ----------

import json
from typing import Any

MODEL_NAME = "databricks-gpt-oss-20b"
DEFAULT_MAX_TOKENS = 1000

RAG_CHAT_ENDPOINT = "databricks-claude-sonnet-4-5"
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"

_openai_client = None


def get_openai_client() -> Any:
    """Initialize an OpenAI-compatible Databricks client only when needed.

    Args:
        None.

    Returns:
        A cached OpenAI-compatible client configured for Databricks.

    Raises:
        RuntimeError: If neither Databricks OpenAI nor the Databricks SDK
            can initialize an authenticated OpenAI client.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        from databricks_openai import DatabricksOpenAI

        _openai_client = DatabricksOpenAI()
        return _openai_client
    except (ImportError, AttributeError, RuntimeError, ValueError) as openai_error:
        try:
            from databricks.sdk import WorkspaceClient

            _openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
            return _openai_client
        except (ImportError, AttributeError, RuntimeError, ValueError) as sdk_error:
            raise RuntimeError(
                "Unable to initialize an OpenAI-compatible Databricks client. "
                "Install databricks-openai, or ensure databricks-sdk and openai "
                "are available in the notebook environment. Also check "
                "workspace authentication. "
                f"Databricks OpenAI error: {openai_error}. "
                f"SDK fallback error: {sdk_error}."
            ) from sdk_error


class _LazyOpenAIClient:
    """Keep the existing client variable without importing optional packages."""

    def __getattr__(self, name: str) -> Any:
        """Access an attribute on the underlying client, initializing lazily.

        Args:
            name: Name of the OpenAI-compatible client attribute.

        Returns:
            The corresponding client attribute.

        Raises:
            RuntimeError: If client initialization fails.
            AttributeError: If the attribute does not exist.
        """
        return getattr(get_openai_client(), name)


client = _LazyOpenAIClient()

# COMMAND ----------

def ask_llm(
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send chat messages to a Databricks Foundation Model API.

    Args:
        messages: Ordered messages containing role and content fields.
        response_format: Optional format contract for structured output.

    Returns:
        The assistant's response as a nonempty string.

    Raises:
        ValueError: If no usable text was generated.
        RuntimeError: If the selected model is unavailable, the caller lacks
            permissions, or the request is rejected by the endpoint.
    """
    request = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    if response_format is not None:
        request["response_format"] = response_format

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to query '{MODEL_NAME}'. Verify Foundation Model API "
            "availability and CAN QUERY permissions."
        ) from exc

    if not response.choices:
        raise ValueError("The model returned no response choices.")
    choice = response.choices[0]
    if choice.finish_reason in {"length", "content_filter"}:
        raise ValueError(f"The model response is incomplete: {choice.finish_reason}")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned no textual response.")
    return content

# COMMAND ----------

def get_chat_model(endpoint: str = RAG_CHAT_ENDPOINT, temperature: float = 0.0) -> "ChatDatabricks":
    """Create a reusable LangChain chat model for CineAI applications.

    Args:
        endpoint: Databricks chat serving endpoint name.
        temperature: Sampling temperature for chat generation.

    Returns:
        A configured ChatDatabricks instance.

    Raises:
        ValueError: If the endpoint name is empty.
        ImportError: If databricks-langchain is unavailable.
        Exception: If the chat model cannot be initialized.
    """
    if not endpoint.strip():
        raise ValueError("Chat endpoint cannot be empty")

    from databricks_langchain import ChatDatabricks

    return ChatDatabricks(endpoint=endpoint, temperature=temperature)


def get_embedding_model(endpoint: str = EMBEDDING_ENDPOINT) -> "DatabricksEmbeddings":
    """Create a reusable LangChain embedding model for retrieval.

    Args:
        endpoint: Databricks embedding serving endpoint name.

    Returns:
        A configured DatabricksEmbeddings instance.

    Raises:
        ValueError: If the endpoint name is empty.
        ImportError: If databricks-langchain is unavailable.
        Exception: If the embedding model cannot be initialized.
    """
    if not endpoint.strip():
        raise ValueError("Embedding endpoint cannot be empty")

    from databricks_langchain import DatabricksEmbeddings

    return DatabricksEmbeddings(endpoint=endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Search · helpers compartidos
# MAGIC
# MAGIC Estas funciones se utilizan en la práctica 4.2 y en el RAG posterior. No crean recursos, no modifican la tabla Delta y no lanzan sincronizaciones.
# MAGIC
# MAGIC **Instalación previa (una vez por entorno):** `databricks-ai-search`. El índice y sus columnas de metadata deben existir previamente.
# MAGIC
# MAGIC **Importante:** CDF se habilita una sola vez después de crear `cineai.rag.document_chunks` (práctica 2.5), no cada vez que se ejecuta este setup.
# MAGIC

# COMMAND ----------

AI_SEARCH_INDEX_NAME = "cineai.rag.cineai_docs_vs"
AI_SEARCH_SOURCE_TABLE = "cineai.rag.document_chunks"
AI_SEARCH_TEXT_COLUMN = "text"
AI_SEARCH_POLICY_DOCUMENT_ID = "CE-CX-POL-001"
AI_SEARCH_PRESS_DOCUMENT_ID = "DOC-PRESS-001"


def get_ai_search_index(index_name: str = AI_SEARCH_INDEX_NAME) -> Any:
    """Connect to an existing CineAI AI Search index.

    Args:
        index_name: Three-part Unity Catalog name of an existing AI Search index.

    Returns:
        AI Search index client that can execute similarity searches.

    Raises:
        ValueError: If the index name is empty.
        ImportError: If the databricks-ai-search package is unavailable.
        RuntimeError: If authentication, permissions, or index lookup fail.
    """
    if not index_name.strip():
        raise ValueError("AI Search index name cannot be empty")

    from databricks.ai_search.client import AISearchClient

    try:
        return AISearchClient().get_index(index_name=index_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to access AI Search index '{index_name}'. "
            "Verify its exact UC name, status, and CAN QUERY permissions."
        ) from exc


def search_ai_search_chunks(
    index: Any,
    question: str,
    document_id: str | None = None,
    num_results: int = 3,
) -> Any:
    """Retrieve relevant CineAI chunks with optional source filtering.

    This helper uses Standard endpoint filter syntax. A Storage Optimized
    endpoint requires a SQL-like filter string instead.

    Args:
        index: Initialized AI Search index client.
        question: Natural-language question to retrieve evidence for.
        document_id: Optional document ID that restricts the source.
        num_results: Maximum number of retrieved chunks.

    Returns:
        pandas.DataFrame: Ranked chunks with IDs, excerpts, and optional score.

    Raises:
        ValueError: If the question is empty or num_results is not positive.
        RuntimeError: If search fails or the index lacks requested columns.
    """
    import pandas as pd

    if not question.strip():
        raise ValueError("Question must not be empty")
    if num_results < 1:
        raise ValueError("num_results must be positive")

    options = {
        "query_text": question,
        "columns": ["chunk_id", "document_id", AI_SEARCH_TEXT_COLUMN],
        "num_results": num_results,
        "query_type": "ANN",
    }
    if document_id is not None:
        options["filters"] = {"document_id": document_id}

    try:
        response = index.similarity_search(**options)
    except Exception as exc:
        raise RuntimeError(
            "AI Search query failed. Verify index readiness, embedding support, "
            f"and that chunk_id, document_id, and {AI_SEARCH_TEXT_COLUMN} "
            f"were indexed. Original error: {exc}"
        ) from exc

    manifest_columns = response.get("manifest", {}).get("columns", [])
    column_names = [entry["name"] for entry in manifest_columns]
    data_array = response.get("result", {}).get("data_array", [])

    if data_array and not column_names:
        raise RuntimeError("The AI Search response has rows but no column manifest")

    output_rows = []
    for rank, values in enumerate(data_array, start=1):
        item = dict(zip(column_names, values))
        output_rows.append({
            "rank": rank,
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "content": item.get(AI_SEARCH_TEXT_COLUMN, ""),
            "score": item.get("score", item.get("_score")),
        })

    return pd.DataFrame(
        output_rows,
        columns=["rank", "chunk_id", "document_id", "content", "score"],
    )
