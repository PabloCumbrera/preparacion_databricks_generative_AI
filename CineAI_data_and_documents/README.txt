CINEAI - GENERATIVE AI ENGINEER ASSOCIATE COURSE MATERIALS
=======================================================

All content is synthetic and created for training. No real customer data is included.

FILES
-----
- movies.csv: Structured catalogue: model tasks, tools, filtering, batch inference.
- screenings.csv: Time-sensitive structured facts: tool calling, agent routing, evaluation.
- ticket_sales.csv: Transactions + synthetic PII: tools, masking, governance, authorization.
- reviews.csv: Unstructured short text: summarization, classification, cleaning, prompt-injection examples.
- cinema_policies.pdf: Authoritative RAG source: extraction, chunking, retrieval, reranking, citations.
- movie_press_kit.docx: Structured document source: section-aware chunking, metadata, factual grounding.
- eclipse_protocol_poster.png: Multimodal input: visual extraction of title, date, director, duration and rating.
- scanned_customer_claim.png: OCR + information extraction + privacy/masking.
- evaluation_set.csv: Offline evaluation: retrieval, tools, safety, structured ground truth and judges.

CANONICAL KEYS
--------------
movie_id -> movies.csv
screening_id -> screenings.csv
ticket_id -> ticket_sales.csv
movie_id links movies, screenings, tickets and reviews.

DESIGN PRINCIPLE
----------------
The files intentionally contain current facts, long-form policies, structured metadata, short reviews, PII-like synthetic fields, untrusted instructions and image text so the same compact dataset can support the full course lifecycle.
