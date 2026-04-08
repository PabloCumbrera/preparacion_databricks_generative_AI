RETAILCO DOCUMENT CORPUS

Contenido:
- Politicas cliente: devoluciones, envios, garantia, terminos.
- FAQs: clientes y facturacion.
- Manuales: portatil y air fryer.
- Interno: playbook de soporte y changelog de catalogo.
- OCR: dos documentos escaneados simulados en PNG.
- Manifest: documents_manifest.csv

Uso recomendado:
1. Subir todos los ficheros a un volumen o carpeta raw.
2. Parsear por tipo de archivo.
3. Aplicar limpieza y chunking.
4. Persistir chunks y metadatos en Delta.
5. Crear indice Vector Search.
6. Evaluar retrieval con retrieval_eval_queries.csv ya creado anteriormente.
