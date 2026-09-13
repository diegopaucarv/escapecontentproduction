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
# Indexar todo lo pendiente/cambiado
python -m src.kag_ingest

# Responder una pregunta
python -m src.kag_query "¿Qué fórmula usa la propagación hacia atrás?"

# Demo completa (indexa + responde, con prints detallados)
python scripts/kag_demo.py --index "tu pregunta aquí"
```
