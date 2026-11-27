# Databricks notebook source
# MAGIC %md
# MAGIC # CineAI · Preparando los documentos
# MAGIC
# MAGIC **Objetivo:** PDF y DOCX → texto → chunks → embeddings → Delta.
# MAGIC
# MAGIC **Entrada:** `/Volumes/cineai/raw/documents/`  
# MAGIC **Salida:** `cineai.rag.document_chunks`

# COMMAND ----------

# MAGIC %md
# MAGIC ![arquitectura_cine_ai_extraccion_limpieza_chunking_embedding.png](./arquitectura_cine_ai_extraccion_limpieza_chunking_embedding.png "arquitectura_cine_ai_extraccion_limpieza_chunking_embedding.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Para leer los documentos. Para el PDF utilizamos PyMuPDF y para el Word, python-docx.

# COMMAND ----------

#%pip install -q pymupdf "python-docx>=1.1.2" databricks-openai

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurar la práctica

# COMMAND ----------

from pathlib import Path
import hashlib
import math
import re

import fitz
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from databricks_openai import DatabricksOpenAI
from pyspark.sql import functions as F, types as T

VOLUME_PATH = Path("/Volumes/cineai/raw/documents")
TABLE_NAME = "cineai.rag.document_chunks"
EMBEDDING_MODEL = "databricks-qwen3-embedding-0-6b"
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 0
EMBEDDING_BATCH_SIZE = 8


def find_source(pattern: str) -> Path:
    matches = sorted(VOLUME_PATH.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern}: {matches}")
    return matches[0]


sources = {
    "policy": find_source("*cinema_policies*.pdf"),
    "press_kit": find_source("*movie_press_kit*.docx"),
}
print({name: str(path) for name, path in sources.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extracción
# MAGIC #### Conservar los bloques, estilos y procedencia sin modificar aún su texto.

# COMMAND ----------


import textwrap


def extract_pdf(path: Path) -> list[dict]:
    blocks = []

    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            for item in page.get_text("blocks", sort=True):
                if item[6] == 0 and item[4].strip():
                    blocks.append({
                        "document_id": "CE-CX-POL-001",
                        "source_uri": str(path),
                        "source_type": "pdf",
                        "source_version": "2026.09",
                        "page_number": page_number,
                        "style": "PDF text",
                        "text": item[4],
                    })

    return blocks


def extract_table_rows(table: Table) -> list[str]:
    rows = []

    for row in table.rows:
        values = []

        for cell in row.cells:
            items = []

            for item in cell.iter_inner_content():
                if isinstance(item, Paragraph) and item.text.strip():
                    items.append(item.text)
                elif isinstance(item, Table):
                    items.extend(extract_table_rows(item))

            value = " ".join(items).strip()

            if value and value not in values:
                values.append(value)

        if values:
            rows.append(" | ".join(values))

    return rows


def extract_docx(path: Path) -> list[dict]:
    blocks = []

    for item in Document(path).iter_inner_content():
        if isinstance(item, Paragraph):
            entries = [(item.style.name, item.text)]
        elif isinstance(item, Table):
            entries = [
                ("Table", row)
                for row in extract_table_rows(item)
            ]
        else:
            continue

        for style, text in entries:
            if text.strip():
                blocks.append({
                    "document_id": "DOC-PRESS-001",
                    "source_uri": str(path),
                    "source_type": "docx",
                    "source_version": "2026-09-03",
                    "page_number": None,
                    "style": style,
                    "text": text,
                })

    return blocks


raw_blocks = (
    extract_pdf(sources["policy"])
    + extract_docx(sources["press_kit"])
)


# Follow the same refund-policy example throughout the notebook
page_blocks = [
    block for block in raw_blocks
    if block["source_type"] == "pdf"
    and block["page_number"] == 3
]

start = next(
    index
    for index, block in enumerate(page_blocks)
    if block["text"].strip() == "1.1 Digital purchases"
)

example_blocks = page_blocks[start:start + 4]

print("=" * 75)
print("EXTRACTION | CINEAI REFUND POLICY | PAGE 3")
print("=" * 75)

print("\n[INCORRECT] FLATTENED DOCUMENT")
print("-" * 75)

flattened_text = re.sub(
    r"\s+",
    " ",
    " ".join(block["text"] for block in example_blocks)
).strip()

print(textwrap.fill(flattened_text, width=95))


print("\n[CORRECT] EXTRACTION PRESERVING SOURCE BLOCKS")
print("-" * 75)

for index, block in enumerate(example_blocks, start=1):
    text = re.sub(r"\s+", " ", block["text"]).strip()

    is_heading = bool(
        re.match(r"^\d+\.\d+\s+", text)
    )

    block_type = "HEADING" if is_heading else "CONTENT"

    print(
        f"\nBLOCK {index} | PAGE {block['page_number']} "
        f"| {block_type}"
    )
    print(textwrap.fill(text, width=95))


print("\n" + "=" * 75)
print(f"TOTAL EXTRACTED BLOCKS: {len(raw_blocks)}")
print("=" * 75)


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## 2. Limpieza
# MAGIC #### Limpiar cabeceras, pies y espacios, preservando los hechos y convirtiendo títulos en metadatos.

# COMMAND ----------


import textwrap


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_pdf(blocks: list[dict]) -> list[dict]:
    cleaned = []
    section = "Document control"
    heading = section
    expect_section_title = False

    for block in blocks:
        text = normalize_text(block["text"])

        if not text or re.fullmatch(
            r"CINEAI\s*\|\s*Customer & Venue Policies", text
        ):
            continue

        if re.match(r"^Policy 2026\.09\s*\|", text):
            continue

        if re.fullmatch(
            r"SECTION \d+|POLICY GOVERNANCE|APPENDIX A", text
        ):
            section = heading = text
            expect_section_title = text.startswith("SECTION ")
            continue

        if expect_section_title:
            section = f"{section} | {text}"
            heading = section
            expect_section_title = False
            continue

        if re.fullmatch(r"\d+\.\d+\s+.+", text):
            heading = text
            continue

        if text in {
            "How to use this manual",
            "Operational decision matrix",
            "DOCUMENT CONTROL",
        }:
            heading = text
            continue

        cleaned.append({
            **{
                key: value for key, value in block.items()
                if key != "style"
            },
            "section": section,
            "heading": heading,
            "movie_id": None,
            "content_class": "official_policy",
            "text": text,
        })

    return cleaned


def clean_docx(blocks: list[dict]) -> list[dict]:
    cleaned = []
    section = heading = "About this press kit"
    movie_id = None
    started = False

    movie_ids = {
        "ECLIPSE PROTOCOL": "MOV001",
        "ORBIT SEVEN": "MOV002",
        "THE QUIET ALGORITHM": "MOV015",
    }

    for block in blocks:
        text = normalize_text(block["text"])
        style = block["style"]

        if style.startswith("Heading 1"):
            section = heading = text
            movie_id = None
            started = True
            continue

        if not started or not text or style in {
            "Kicker", "SmallMuted"
        }:
            continue

        if style.startswith("Heading 2"):
            heading = text
            continue

        if style == "Table":
            for title, identifier in movie_ids.items():
                if title in text:
                    section = title
                    heading = "Film overview"
                    movie_id = identifier
                    break

        cleaned.append({
            **{
                key: value for key, value in block.items()
                if key != "style"
            },
            "section": section,
            "heading": heading,
            "movie_id": movie_id,
            "content_class": (
                "untrusted_example"
                if heading == "Embedded untrusted-content example"
                else "approved_press_material"
            ),
            "text": text,
        })

    return cleaned


source_blocks = clean_pdf([
    block for block in raw_blocks
    if block["source_type"] == "pdf"
])

source_blocks += clean_docx([
    block for block in raw_blocks
    if block["source_type"] == "docx"
])


# Example 1: preserving business information

raw_example = next(
    block["text"]
    for block in raw_blocks
    if "Tickets purchased through" in block["text"]
)

clean_example = next(
    block
    for block in source_blocks
    if "Tickets purchased through" in block["text"]
)

bad_cleaning = re.sub(
    r"\b\d+\b", "",
    normalize_text(raw_example)
)

print("=" * 75)
print("CLEANING | CINEAI REFUND POLICY | SECTION 1.1")
print("=" * 75)

print("\n[ORIGINAL] TEXT BEFORE CLEANING")
print("-" * 75)
print(repr(raw_example))

print("\n[INCORRECT] DESTRUCTIVE CLEANING")
print("-" * 75)
print(textwrap.fill(bad_cleaning, width=95))

print("\nPROBLEM:")
print("  - Numeric values are removed indiscriminately.")
print("  - The cancellation deadline loses its value: 2 hours.")
print("  - A critical business rule becomes incomplete.")


print("\n[CORRECT] CLEANING PRESERVING BUSINESS RULES")
print("-" * 75)
print(textwrap.fill(clean_example["text"], width=95))

print("\nPRESERVED:")
print("  - Deadline: 2 hours before the screening.")
print("  - All original business conditions.")
print("  - Normalized whitespace and line breaks.")


# Example 2: removing repeated document noise

raw_header = next(
    block["text"]
    for block in raw_blocks
    if re.fullmatch(
        r"CINEAI\s*\|\s*Customer & Venue Policies",
        normalize_text(block["text"])
    )
)

header_removed = not any(
    normalize_text(raw_header) == block["text"]
    for block in source_blocks
)

print("\n" + "=" * 75)
print("CLEANING | REMOVING DOCUMENT NOISE")
print("=" * 75)

print("\n[ORIGINAL] REPEATED PDF HEADER")
print(repr(raw_header))

print("\n[CORRECT] HEADER REMOVED")
print("Removed from clean content:", header_removed)

print("\nRESULT:")
print("  - Repeated PDF headers do not enter the RAG corpus.")
print("  - Relevant policy paragraphs remain available.")


# Example 3: preserving structure and source metadata

print("\n" + "=" * 75)
print("CLEANING | PRESERVING DOCUMENT STRUCTURE")
print("=" * 75)

print("\n[CORRECT] STRUCTURED POLICY BLOCK")
print("-" * 75)

for field in [
    "document_id",
    "source_type",
    "page_number",
    "section",
    "heading",
    "source_version",
    "content_class",
]:
    print(f"  {field:18}: {clean_example[field]}")


movie_example = next(
    block
    for block in source_blocks
    if block["movie_id"] == "MOV001"
    and block["heading"] == "Extended synopsis"
)

print("\n[CORRECT] STRUCTURED MOVIE BLOCK")
print("-" * 75)

print("  document_id       :", movie_example["document_id"])
print("  movie_id          :", movie_example["movie_id"])
print("  section           :", movie_example["section"])
print("  heading           :", movie_example["heading"])
print("  content_class     :", movie_example["content_class"])

print("\nRESULT:")
print("  - Policy paragraphs retain section and subsection.")
print("  - Movie paragraphs retain their movie identifier.")
print("  - Source provenance remains available for citations.")


# Validate cleaning rules

checks = {
    "Repeated PDF headers removed": header_removed,
    "Whitespace normalized": all(
        b["text"] == normalize_text(b["text"])
        for b in source_blocks
    ),
    "Two-hour deadline preserved": (
        "2 hours" in clean_example["text"]
    ),
    "Policy subsection retained": (
        clean_example["heading"] == "1.1 Digital purchases"
    ),
    "Three movies identified": {
        b["movie_id"]
        for b in source_blocks
        if b["movie_id"]
    } == {"MOV001", "MOV002", "MOV015"},
    "Untrusted example flagged": any(
        b["content_class"] == "untrusted_example"
        for b in source_blocks
    ),
    "Source provenance retained": all(
        b["document_id"]
        and b["source_uri"]
        and b["source_version"]
        for b in source_blocks
    ),
}

print("\n" + "=" * 75)
print("CLEANING VALIDATION")
print("=" * 75)

for rule, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'} | {rule}")

assert all(checks.values()), "Cleaning validation failed"

print("\n" + "=" * 75)
print("CLEANING SUMMARY")
print("=" * 75)
print(f"  Original blocks : {len(raw_blocks)}")
print(f"  Clean blocks    : {len(source_blocks)}")
print(f"  Reduction       : {len(raw_blocks) - len(source_blocks)}")
print("\nReady for chunking: source_blocks")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chunking
# MAGIC #### Regla: un chunk no cruza documento, página, sección, subsección ni clase de contenido. Se mantiene el encabezado; tamaño completo ≤1.200 caracteres; overlap = 0.

# COMMAND ----------


import textwrap


def split_long_text(text: str, limit: int) -> list[str]:
    units = []

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) <= limit:
            units.append(sentence)
            continue

        part = ""

        for word in sentence.split():
            candidate = f"{part} {word}".strip()

            if len(candidate) > limit and part:
                units.append(part)
                part = word
            else:
                part = candidate

        if part:
            units.append(part)

    return units


def build_chunks(blocks: list[dict], max_chars: int) -> list[dict]:
    chunks = []
    positions = {}

    key_fields = (
        "document_id",
        "source_version",
        "source_uri",
        "section",
        "heading",
        "page_number",
        "content_class",
        "movie_id",
    )

    current_key = None
    current_meta = None
    current_parts = []

    def flush() -> None:
        nonlocal current_parts

        if not current_parts:
            return

        prefix = " | ".join(
            dict.fromkeys((
                current_meta["section"],
                current_meta["heading"],
            ))
        )

        text = f"{prefix}\n{' '.join(current_parts)}"

        document_id = current_meta["document_id"]
        position = positions.get(document_id, 0)
        positions[document_id] = position + 1

        chunk_id = hashlib.sha256(
            (
                f"{document_id}|"
                f"{current_meta['source_version']}|"
                f"{position}|{text}"
            ).encode("utf-8")
        ).hexdigest()[:24]

        chunks.append({
            **{
                key: value
                for key, value in current_meta.items()
                if key != "text"
            },
            "chunk_id": chunk_id,
            "chunk_position": position,
            "text": text,
        })

        current_parts = []

    for block in blocks:
        key = tuple(block[field] for field in key_fields)

        if current_parts and key != current_key:
            flush()

        current_key = key
        current_meta = block

        prefix = " | ".join(
            dict.fromkeys((
                block["section"],
                block["heading"],
            ))
        )

        body_limit = max_chars - len(prefix) - 1

        if body_limit < 100:
            raise ValueError(
                f"Chunk header is too long: {prefix}"
            )

        for unit in split_long_text(block["text"], body_limit):
            candidate = " ".join(current_parts + [unit])

            if current_parts and len(candidate) > body_limit:
                flush()

            current_parts.append(unit)

    flush()
    return chunks


chunks = build_chunks(source_blocks, MAX_CHUNK_CHARS)

digital = next(
    chunk for chunk in chunks
    if chunk["document_id"] == "CE-CX-POL-001"
    and chunk["heading"] == "1.1 Digital purchases"
)

box_office = next(
    chunk for chunk in chunks
    if chunk["document_id"] == "CE-CX-POL-001"
    and chunk["heading"] == "1.2 Box-office purchases"
)


def chunk_body(chunk: dict) -> str:
    return chunk["text"].split("\n", 1)[1]


def show_chunk(chunk: dict, label: str) -> None:
    print(f"\n{label}")
    print("-" * 75)
    print(f"  Document       : {chunk['document_id']}")
    print(f"  Section        : {chunk['section']}")
    print(f"  Subsection     : {chunk['heading']}")
    print(f"  Page           : {chunk['page_number']}")
    print(f"  Size           : {len(chunk['text'])} / {MAX_CHUNK_CHARS} chars")
    print(f"  Chunk ID       : {chunk['chunk_id']}")

    print("\nCONTENT:")
    print(textwrap.fill(chunk["text"], width=95))


print("=" * 75)
print("CHUNKING | CINEAI REFUND POLICY | PAGE 3")
print("=" * 75)

print("\nSTRATEGY CONFIGURATION")
print("-" * 75)
print("  Strategy       : Document structure")
print(f"  Max size       : {MAX_CHUNK_CHARS} characters")
print(f"  Overlap        : {OVERLAP_CHARS} characters")
print("  Boundaries     : Section, subsection and page")
print("  Long content   : Split into sentences when needed")
print("  Chunk context  : Section + subsection repeated in header")


# Demonstration of mixing two independent business rules

bad_flattened = (
    f"{digital['heading']}: {chunk_body(digital)} "
    f"{box_office['heading']}: {chunk_body(box_office)}"
)

print("\n" + "=" * 75)
print("EXAMPLE 1 | WHY DOCUMENT STRUCTURE MATTERS")
print("=" * 75)

print("\n[INCORRECT] TWO DIFFERENT RULES MIXED INTO ONE TEXT")
print("-" * 75)
print(textwrap.fill(bad_flattened, width=95))

print("\nPROBLEM:")
print("  - Digital and box-office purchase rules are mixed.")
print("  - Both subsections address different business conditions.")
print("  - A single fragment would contain two different rules.")
print(
    f"  - Combined size: {len(bad_flattened)} characters."
)
print("  - Meeting the size limit does not guarantee a coherent chunk.")


# Show the two independent chunks generated by the actual strategy

print("\n" + "=" * 75)
print("EXAMPLE 2 | STRUCTURE-AWARE CHUNKING")
print("=" * 75)

show_chunk(digital, "[CORRECT] CHUNK A - DIGITAL PURCHASES")
show_chunk(box_office, "[CORRECT] CHUNK B - BOX-OFFICE PURCHASES")

print("\nRESULT:")
print("  - Each subsection generates independent content.")
print("  - The section and subsection are preserved.")
print("  - Each chunk retains its document and page.")
print("  - No body text is copied from one chunk into the next.")
print("  - Repeated metadata headers are not text overlap.")


# Explain why the first chunk does not fill its maximum capacity

print("\n" + "=" * 75)
print("EXAMPLE 3 | CHUNK SIZE AND OVERLAP")
print("=" * 75)

used_chars = len(digital["text"])
remaining_chars = MAX_CHUNK_CHARS - used_chars

print(f"\n  Maximum capacity : {MAX_CHUNK_CHARS} characters")
print(f"  Actual chunk size: {used_chars} characters")
print(f"  Unused capacity  : {remaining_chars} characters")
print(f"  Overlap          : {OVERLAP_CHARS} characters")

print("\nWHY DOES THE CHUNK STOP HERE?")
print("  - The next subsection begins: 1.2 Box-office purchases.")
print("  - We respect the subsection boundary.")
print("  - We do not add unrelated content to fill 1200 characters.")
print("  - No overlap is applied between content fragments.")


# Check that the full subsection is retained exactly once

digital_source_text = " ".join(
    block["text"]
    for block in source_blocks
    if block["document_id"] == "CE-CX-POL-001"
    and block["heading"] == "1.1 Digital purchases"
)

digital_chunk_text = " ".join(
    chunk_body(chunk)
    for chunk in chunks
    if chunk["document_id"] == "CE-CX-POL-001"
    and chunk["heading"] == "1.1 Digital purchases"
)

checks = {
    "Maximum chunk size respected": all(
        len(chunk["text"]) <= MAX_CHUNK_CHARS
        for chunk in chunks
    ),
    "Document metadata retained": all(
        chunk["document_id"]
        and chunk["section"]
        and chunk["heading"]
        for chunk in chunks
    ),
    "Digital purchase rule preserved": (
        "2 hours before" in digital["text"]
    ),
    "Different subsections separated": (
        digital["heading"] != box_office["heading"]
        and box_office["heading"] not in digital["text"]
        and digital["heading"] not in box_office["text"]
    ),
    "Original subsection content preserved": (
        digital_source_text == digital_chunk_text
    ),
    "Unique chunk identifiers": (
        len({chunk["chunk_id"] for chunk in chunks})
        == len(chunks)
    ),
    "Zero configured content overlap": (
        OVERLAP_CHARS == 0
    ),
}

print("\n" + "=" * 75)
print("CHUNKING VALIDATION")
print("=" * 75)

for rule, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'} | {rule}")

assert all(checks.values()), "Chunking validation failed"


print("\n" + "=" * 75)
print("CHUNKING SUMMARY")
print("=" * 75)

sizes = [len(chunk["text"]) for chunk in chunks]

print(f"  Input blocks      : {len(source_blocks)}")
print(f"  Generated chunks  : {len(chunks)}")
print(f"  Smallest chunk    : {min(sizes)} characters")
print(f"  Largest chunk     : {max(sizes)} characters")
print(f"  Maximum allowed   : {MAX_CHUNK_CHARS} characters")
print(f"  Overlap           : {OVERLAP_CHARS} characters")
print("\nReady for embeddings: chunks")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generar embeddings
# MAGIC #### Vectorizar el chunk con encabezados y contenido. Las preguntas deben utilizar el mismo modelo.

# COMMAND ----------


import textwrap


# Follow the same CineAI refund policy example
bad_input = digital["heading"]
good_input = digital["text"]

question = (
    "¿Con cuánta antelación puedo cancelar "
    "una entrada digital de CineAI?"
)


print("=" * 75)
print("EMBEDDINGS | CINEAI REFUND POLICY | SECTION 1.1")
print("=" * 75)

print("\n" + "=" * 75)
print("EXAMPLE 1 | WHAT TEXT SHOULD WE VECTORIZE?")
print("=" * 75)

print("\n[INCORRECT] EMBEDDING ONLY THE SUBSECTION TITLE")
print("-" * 75)
print(bad_input)

print("\nPROBLEM:")
print("  - The title identifies the topic: digital purchases.")
print("  - It does not contain the cancellation conditions.")
print("  - The critical deadline of 2 hours is missing.")
print("  - This is a demonstration; this input is not vectorized.")


print("\n[CORRECT] EMBEDDING THE COMPLETE CHUNK")
print("-" * 75)
print(textwrap.fill(good_input, width=95))

print("\nPRESERVED:")
print("  - Section and subsection provide context.")
print("  - The full cancellation rule is included.")
print("  - The 2-hour deadline remains in the text.")
print("  - The original text is retained alongside the vector.")


# Generate embeddings for every document chunk

print("\n" + "=" * 75)
print("EXAMPLE 2 | CONVERTING DOCUMENT TEXT INTO VECTORS")
print("=" * 75)

client = DatabricksOpenAI()

for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
    batch = chunks[start:start + EMBEDDING_BATCH_SIZE]

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[chunk["text"] for chunk in batch],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding request failed for {EMBEDDING_MODEL}. "
            "Check endpoint access and CAN QUERY permissions."
        ) from exc

    vectors = sorted(
        response.data,
        key=lambda item: item.index
    )

    if len(vectors) != len(batch):
        raise ValueError(
            "The embedding endpoint returned an incomplete batch"
        )

    for chunk, vector in zip(batch, vectors):
        chunk["embedding"] = [
            float(value) for value in vector.embedding
        ]
        chunk["embedding_model"] = EMBEDDING_MODEL


document_vector = digital["embedding"]
dimensions = len(document_vector)

print(f"\n  Embedding model : {EMBEDDING_MODEL}")
print(f"  Chunks encoded  : {len(chunks)}")
print(f"  Input characters: {len(good_input)}")
print(f"  Vector dimensions: {dimensions}")

print("\nDOCUMENT VECTOR (FIRST 8 VALUES):")
print([round(value, 5) for value in document_vector[:8]])

print("\nWHAT DOES THIS MEAN?")
print(f"  - One text chunk becomes one vector of {dimensions} numbers.")
print("  - The numbers represent a learned semantic representation.")
print("  - Individual vector values are not directly interpretable.")
print("  - The original text is still needed to recover exact facts.")


# Convert the user question into a vector with the same model

print("\n" + "=" * 75)
print("EXAMPLE 3 | VECTORIZING A USER QUESTION")
print("=" * 75)

print("\nUSER QUESTION:")
print(question)

try:
    question_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
except Exception as exc:
    raise RuntimeError(
        f"Question embedding failed for {EMBEDDING_MODEL}"
    ) from exc

question_vector = [
    float(value)
    for value in question_response.data[0].embedding
]

print("\nQUESTION VECTOR (FIRST 8 VALUES):")
print([round(value, 5) for value in question_vector[:8]])

print("\nVECTOR COMPATIBILITY:")
print(f"  Document model     : {EMBEDDING_MODEL}")
print(f"  Question model     : {EMBEDDING_MODEL}")
print(f"  Document dimensions: {len(document_vector)}")
print(f"  Question dimensions: {len(question_vector)}")

print("\nRESULT:")
print("  - Documents and questions use the same embedding model.")
print("  - Their vectors belong to the same embedding space.")
print("  - They can be compared using vector similarity.")
print("  - Similarity search and retrieval will be implemented later.")


# Validate document and question embeddings

vector_dimensions = {
    len(chunk["embedding"])
    for chunk in chunks
}

checks = {
    "Complete chunk used as embedding input": (
        good_input == digital["text"]
    ),
    "Critical business deadline preserved": (
        "2 hours before" in good_input
    ),
    "All chunks have an embedding": all(
        "embedding" in chunk and chunk["embedding"]
        for chunk in chunks
    ),
    "Consistent document vector dimensions": (
        len(vector_dimensions) == 1
        and dimensions > 0
    ),
    "Question and document dimensions match": (
        len(question_vector) == dimensions
    ),
    "Same embedding model used": all(
        chunk["embedding_model"] == EMBEDDING_MODEL
        for chunk in chunks
    ),
    "All embedding values are finite": all(
        math.isfinite(value)
        for chunk in chunks
        for value in chunk["embedding"]
    ) and all(
        math.isfinite(value)
        for value in question_vector
    ),
}

print("\n" + "=" * 75)
print("EMBEDDING VALIDATION")
print("=" * 75)

for rule, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'} | {rule}")

assert all(checks.values()), "Embedding validation failed"


print("\n" + "=" * 75)
print("EMBEDDING SUMMARY")
print("=" * 75)

print(f"  Embedding model      : {EMBEDDING_MODEL}")
print(f"  Vectorized chunks    : {len(chunks)}")
print(f"  Embedding dimensions : {dimensions}")
print(f"  Example question     : {question}")
print("  Original text        : Preserved")
print("  Chunk metadata       : Preserved")
print("\nReady for Delta: chunks")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Guardar chunks y vectores en Delta

# COMMAND ----------

schema = T.StructType([
    T.StructField("chunk_id", T.StringType(), False),
    T.StructField("document_id", T.StringType(), False),
    T.StructField("chunk_position", T.IntegerType(), False),
    T.StructField("source_uri", T.StringType(), False),
    T.StructField("source_type", T.StringType(), False),
    T.StructField("source_version", T.StringType(), False),
    T.StructField("section", T.StringType(), False),
    T.StructField("heading", T.StringType(), False),
    T.StructField("movie_id", T.StringType(), True),
    T.StructField("page_number", T.IntegerType(), True),
    T.StructField("content_class", T.StringType(), False),
    T.StructField("text", T.StringType(), False),
    T.StructField("embedding_model", T.StringType(), False),
    T.StructField("embedding", T.ArrayType(T.DoubleType(), False), False),
])

spark.sql("CREATE SCHEMA IF NOT EXISTS cineai.rag")
spark.createDataFrame(chunks, schema=schema).write.format("delta") \
    .mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)
spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Saved {len(chunks)} vectorized chunks to {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Comprobar el resultado

# COMMAND ----------

stored = spark.table(TABLE_NAME)

stored.groupBy("source_type", "document_id").agg(
    F.count("*").alias("chunks"),
    F.min(F.length("text")).alias("min_chars"),
    F.max(F.length("text")).alias("max_chars"),
    F.min(F.size("embedding")).alias("min_dimensions"),
    F.max(F.size("embedding")).alias("max_dimensions"),
).display()

stored.filter(F.col("heading") == "1.1 Digital purchases").select(
    "document_id", "section", "heading", "page_number", "text",
    F.length("text").alias("chars"),
    "embedding_model", F.size("embedding").alias("dimensions"),
).display()

assert stored.filter(F.length("text") > MAX_CHUNK_CHARS).count() == 0
assert stored.filter(F.col("embedding").isNull()).count() == 0
assert stored.select("chunk_id").distinct().count() == stored.count()
print("PASS: complete trace, chunk limits, and vectors persisted")