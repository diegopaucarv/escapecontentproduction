"""Asegura que los modelos Stanza de español estén descargados (con reintentos).

El segmentador llama a stanza.download('es', processors='tokenize,pos,lemma,
depparse,constituency,coref') en get_stanza. Esa llamada re-descarga
resources.json desde GitHub en cada invocación y falla con ConnectionResetError
si la red es inestable. Este script reintenta hasta completar.
"""

from __future__ import annotations

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import stanza

PROCESSORS = "tokenize,pos,lemma,depparse,constituency,coref"
MAX_ATTEMPTS = 8

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(
        f"[stanza] intento {attempt}/{MAX_ATTEMPTS} — descargando es ({PROCESSORS})...",
        flush=True,
    )
    try:
        stanza.download("es", processors=PROCESSORS, verbose=True)
        print("[stanza] ✅ descarga completada", flush=True)
        break
    except Exception as exc:  # noqa: BLE001
        print(
            f"[stanza] ❌ intento {attempt} falló: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if attempt == MAX_ATTEMPTS:
            raise
        time.sleep(5)

# Verificar que el pipeline carga (con el parche del segmentador aplicado).
print("[stanza] verificando carga del pipeline...", flush=True)
t0 = time.time()
pipe = stanza.Pipeline(
    "es",
    processors=PROCESSORS,
    use_gpu=False,
    verbose=False,
)
print(f"[stanza] ✅ pipeline cargado en {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
doc = pipe("El perro corre. El perro ladra.")
print(
    f"[stanza] ✅ inferencia en {time.time() - t0:.0f}s, oraciones: {len(doc.sentences)}",
    flush=True,
)
