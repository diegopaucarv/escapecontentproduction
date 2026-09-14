"""Parche para el bug de Stanza coref (Config.__init__ exige plateau_epochs).

Stanza 1.11.x añadió `plateau_epochs` como argumento requerido de
`Config.__init__`, pero el loader de modelos (`CorefModel.load_model`)
lo construye con `Config(**config)` sin pasarlo → TypeError. El parche
inyecta el default en `__init__`.

Idempotente: se puede llamar varias veces (flag en la clase). Lo usan
`src/kag/segmentador.py` (a nivel de módulo) y `scripts/ensure_stanza.py`
(que crea un pipeline Stanza sin importar el segmentador).
"""

from __future__ import annotations


def apply_stanza_coref_patch() -> None:
    try:
        from stanza.models.coref.config import Config as _StanzaCorefCfg

        if not hasattr(_StanzaCorefCfg, "_coref_patched_flag"):
            _StanzaCorefCfg._coref_original_init = _StanzaCorefCfg.__init__

            def _coref_patched_init(self, *args, **kwargs):
                kwargs.setdefault("plateau_epochs", 10)
                _StanzaCorefCfg._coref_original_init(self, *args, **kwargs)

            _StanzaCorefCfg.__init__ = _coref_patched_init
            _StanzaCorefCfg._coref_patched_flag = True
            print("[PATCH] Stanza coref Config parcheado correctamente.")
    except ImportError:
        pass
