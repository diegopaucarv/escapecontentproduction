# Knowledge Repository — KAG

Este directorio es la fuente de conocimiento del sistema KAG
(`src/kag_ingest.py` + `src/kag_query.py`, ver `docs/diseno_sistema_kag.md`).

## Estructura esperada

```
data/knowledge_repository/
├── docs/                          # los .md (artículos, book stacks)
│   ├── paper_nanotech.md
│   └── book_stack_deep_learning.md
└── images/
    └── paper_nanotech/            # carpeta por documento (nombre = .md sin extensión)
        ├── fig1_architecture.jpg
        └── chart_benchmark.png
```

## Reglas

1. **Los `.md` van en `docs/`.** Pueden ser artículos cortos (<20k tokens) o
   book stacks masivos (hasta 500k+ tokens). El sistema los chunkifica solo.
2. **Las imágenes van en `images/[nombre_del_md_sin_extensión]/`.**
   - Ejemplo: `docs/paper_nanotech.md` → `images/paper_nanotech/fig1.jpg`.
   - Formatos aceptados: `.jpg`, `.jpeg`, `.png`.
   - Si la carpeta de imágenes de un documento no existe, la ingesta de
     figuras se omite silenciosamente (no es un error).
3. **Re-indexar:** el sistema es idempotente por hash del contenido. Si
   editas un `.md`, la próxima ingesta lo re-indexa solo. Para forzar:
   `python -m src.kag_ingest --force`.

## Uso

```bash
# 1. Configurar el sistema KAG (segmentador + modelo local + embeddings)
python -m src.db.seed_kag

# 2. Indexar todo lo pendiente/cambiado
python -m src.kag_ingest

# 3. Responder una pregunta
python -m src.kag_query "¿Qué fórmula usa la propagación hacia atrás?"

# Demo completa (indexa + responde, con prints detallados)
python scripts/kag_demo.py --index "tu pregunta aquí"
```

> **Embeddings:** `seed_kag` registra el modelo `jinaai/jina-embeddings-v5-text-nano`
> y crea la `embedding_settings` activa. El token de HuggingFace se lee de
> `HUGGINGFACE_API_KEY` o `HF_TOKEN` (opcional: el modelo es público). Si un
> documento se indexó sin embeddings (chunks con `embedding = NULL`), re-indexa
> con `--force` para poblar los vectores: `python -m src.kag_ingest --force`.
> Sin `embedding_settings`, la consulta degrada a solo FTS (sigue respondiendo,
> pero sin búsqueda densa).
