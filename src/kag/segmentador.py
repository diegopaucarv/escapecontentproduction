import gc
import json
import logging
import re
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import hnswlib
import numpy as np
import spacy
import stanza
import torch
from rapidfuzz import fuzz
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from sqlalchemy import text
from transformers import AutoTokenizer, BertModel, BertTokenizer, pipeline


# ── Optimización de recursos: hilos de torch ─────────────────────────────────
# Los modelos de embeddings del segmentador son pequeños (all-MiniLM, jina
# nano). torch usa por defecto TODOS los cores, lo que en modelos pequeños
# causa thread thrashing (overhead de sincronización > ganancia). Se limita a
# un número sensato de hilos, configurable vía KAG_TORCH_THREADS. No cambia el
# resultado numérico (mismos pesos, mismos vectores), solo el rendimiento.
def _configure_torch_threads() -> None:
    import os

    try:
        if os.environ.get("KAG_TORCH_THREADS"):
            n = max(1, int(os.environ["KAG_TORCH_THREADS"]))
        else:
            cpus = os.cpu_count() or 4
            # Cap: para modelos pequeños, >4 hilos casi nunca ayuda y añade
            # overhead. En GPU los hilos CPU apenas importan (el cómputo va
            # al GPU); en CPU 2-4 hilos suele ser el punto dulce.
            n = min(4, max(1, cpus))
        torch.set_num_threads(n)
        logging.info(f"[Threads] torch.set_num_threads({n})")
    except Exception:  # noqa: BLE001 — nunca debe romper la carga
        pass


_configure_torch_threads()


# ── Caché de embeddings acotada (LRU) ────────────────────────────────────────
# Por qué existe: build_segmenter() cachea el ProgressiveSegmenter por idioma
# (ver _SEGMENTER_CACHE al final del archivo) para no recargar spaCy/
# SentenceTransformer/NLI en cada documento. Eso significa que _embed_cache
# vive mientras vive el proceso, no un solo documento — y acumula entradas de
# TODOS los documentos que pasen por ese worker.
#
# La mayoría de esas entradas son concatenaciones de segmento (crecen en cada
# merge — ver ClassicSegmenter.segment_sentences y
# ProgressiveSegmenter.progressive_clustering) que casi nunca se repiten
# verbatim entre documentos distintos. Sin límite, un dict normal crece para
# siempre. Esta clase es un dict con límite de tamaño y desalojo LRU.
#
# Importante: esto NUNCA cambia un resultado de segmentación. encode() es
# determinístico — texto -> vector es siempre el mismo, así que un miss por
# desalojo simplemente vuelve a calcular el vector idéntico. Lo único que
# varía con el tamaño de la caché es memoria y velocidad, nunca la salida.
class _BoundedEmbeddingCache:
    """Dict-like caché texto -> vector con tope de tamaño (LRU)."""

    def __init__(self, max_size: int = 20_000):
        self.max_size = max_size
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        with self._lock:
            self._data.move_to_end(key)
            return self._data[key]

    def __setitem__(self, key, value) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)  # descarta el menos usado

    def __len__(self) -> int:
        return len(self._data)


def _default_embed_cache_size() -> int:
    """Tope por defecto del caché de embeddings, configurable vía
    KAG_EMBED_CACHE_SIZE. Cada vector pesa unos pocos KB (384 floats para
    all-MiniLM), así que 20k entradas son ~30MB — barato frente al riesgo de
    una caché sin límite en un proceso de larga vida."""
    import os

    try:
        return max(1, int(os.environ.get("KAG_EMBED_CACHE_SIZE", 20_000)))
    except Exception:  # noqa: BLE001 — valor mal formado: usa el default
        return 20_000


# ── Configuración por defecto del segmentador ────────────────────────────────
# La configuración real se lee de kag_segmenter_settings en la DB (ver
# load_segmenter_config); estos defaults solo existen para entornos sin DB.
DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
DEFAULT_SEGMENTER_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPACY_MODEL = "es_core_news_md"
DEFAULT_SPACY_MODELS = {
    "es": "es_core_news_md",
    "en": "en_core_web_md",
    "pt": "pt_core_news_md",
    "de": "de_core_news_md",
    "fr": "fr_core_news_md",
}

# ── Stanza coref bug-fix (compartido con scripts/ensure_stanza.py) ──────────
from src.kag.stanza_patch import apply_stanza_coref_patch

apply_stanza_coref_patch()

# ── Idiomas y procesadores de Stanza (compartido con ensure_languages.py) ────
from src.kag.langs import (
    STANZA_COREF_LANGS,
    stanza_processors,
)

# ── Pivots conversacionales multilingües (archivo externo) ────────────────────
# Los pivots por idioma viven en src/kag/pivots.json. Si se añade un idioma,
# actualízalo con scripts/update_pivots.py (o edita el JSON a mano).
_PIVOTS_PATH = Path(__file__).resolve().parent / "pivots.json"
_PIVOTS_CACHE: dict[str, set[str]] = {}


def _load_pivots(lang: str) -> set[str]:
    """Devuelve los pivots conversacionales del idioma (cacheado)."""
    if lang in _PIVOTS_CACHE:
        return _PIVOTS_CACHE[lang]
    try:
        data = json.loads(_PIVOTS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — archivo ausente/corrupto: fallback vacío
        data = {}
    pivots = set(data.get(lang, []))
    _PIVOTS_CACHE[lang] = pivots
    return pivots


@Language.component("conversational_sbd")
def conversational_sbd(doc):
    # Pivots conversacionales del idioma del documento (multilingüe).
    pivots = _load_pivots(doc.lang_)
    if not pivots:
        return doc

    # We iterate through the tokens, looking for specific syntactic patterns
    for i in range(len(doc) - 2):
        token = doc[i]
        if (
            token.lower_ in pivots
            and doc[i + 1].text == ","
            and doc[i + 2].pos_ == "VERB"
        ):
            doc[i].is_sent_start = True
        elif token.lower_ in {"bueno", "o sea", "entonces"}:
            doc[i].is_sent_start = True
    return doc


device = 0 if torch.cuda.is_available() else -1


# ─────────────────────────────────────────────────────────────────────────────
# AttentionShiftDetector  (FIXED: self.device now assigned in __init__)
# ─────────────────────────────────────────────────────────────────────────────
class AttentionShiftDetector:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, attn_implementation="eager")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # FIX: was missing
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 512

    def get_attention_weights(self, text):
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("[ASD] CUDA not available, falling back to CPU")
            self.device = "cpu"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        attentions = []
        num_chunks = (len(inputs["input_ids"][0]) // self.max_length) + 1
        for i in range(num_chunks):
            chunk_input = inputs["input_ids"][
                :, i * self.max_length : (i + 1) * self.max_length
            ]
            if chunk_input.size(1) == 0:
                continue
            chunk_attention = self.model(chunk_input, output_attentions=True)
            attentions.append(chunk_attention.attentions)
            del chunk_attention
            torch.cuda.empty_cache()
        attentions = [torch.cat(att, dim=1) for att in zip(*attentions)]
        return attentions

    def compare_attention(self, attentions1, attentions2):
        attention_diff = 0
        min_len = min(len(attentions1), len(attentions2))
        for layer1, layer2 in zip(attentions1[:min_len], attentions2[:min_len]):
            avg_attention1 = layer1.mean(dim=1).squeeze().detach().cpu().numpy()
            avg_attention2 = layer2.mean(dim=1).squeeze().detach().cpu().numpy()
            if avg_attention1.shape != avg_attention2.shape:
                max_len = max(avg_attention1.shape[0], avg_attention2.shape[0])
                avg_attention1 = np.pad(
                    avg_attention1, (0, max_len - avg_attention1.shape[0])
                )
                avg_attention2 = np.pad(
                    avg_attention2, (0, max_len - avg_attention2.shape[0])
                )
            attention_diff += np.abs(avg_attention1 - avg_attention2).mean()
        return attention_diff

    def detect_topic_shift(self, segment1, segment2, threshold=0.5):
        attentions1 = self.get_attention_weights(segment1)
        attentions2 = self.get_attention_weights(segment2)
        attention_diff = self.compare_attention(attentions1, attentions2)

        return attention_diff > threshold


# ─────────────────────────────────────────────────────────────────────────────
# ClassicSegmenter — la lógica de segmentación es la original; usa la caché
# de embeddings compartida y acotada de ProgressiveSegmenter (ver
# _BoundedEmbeddingCache) cuando se instancia desde ahí.
# ─────────────────────────────────────────────────────────────────────────────
class ClassicSegmenter:
    def __init__(
        self,
        embedding_model,
        spacy_model,
        nli_model=DEFAULT_NLI_MODEL,
        embed_cache=None,
    ):
        self.nlp = spacy.load(spacy_model)
        self.embedding_model = embedding_model
        # Caché compartida con el ProgressiveSegmenter (si se pasa): mismo
        # texto -> mismo vector, solo evita re-embeder lo ya embebido. Si se
        # instancia suelta (sin ProgressiveSegmenter), usa su propia caché
        # acotada en vez de un dict sin límite.
        self.embedding_cache = (
            embed_cache if embed_cache is not None else _BoundedEmbeddingCache()
        )
        self.nli_model_name = nli_model
        self._nli_model = None  # carga perezosa (~1.6GB, solo si hace falta)

    def semantic_cohesion_score(self, segment1, segment2):
        doc1 = self.nlp(segment1)
        doc2 = self.nlp(segment2)
        ents1 = {ent.text for ent in doc1.ents}
        ents2 = {ent.text for ent in doc2.ents}
        if not (ents1 or ents2):
            return 0.0
        intersection = ents1 & ents2
        union = ents1 | ents2
        return len(intersection) / len(union)

    def compute_semantic_shift(self, segment1, segment2):
        try:
            if not segment1 or not segment2:
                return 1.0
            if self._nli_model is None:
                self._nli_model = pipeline(
                    "zero-shot-classification",
                    model=self.nli_model_name,
                    device=device,
                )
            seg1_tail = segment1[-1000:] if len(segment1) > 1000 else segment1
            result = self._nli_model(seg1_tail, candidate_labels=[segment2])
            return result["scores"][0] if "scores" in result else 1.0
        except Exception as e:
            print(f"[ClassicSeg] Error computing semantic shift: {e}")
            return 1.0

    def compute_boundary_score(self, segment1, segment2):
        # ── NEW: Substance Filter for Overflow Cuts ──
        doc2 = self.nlp(segment2)
        content_pos = {"VERB", "NOUN", "ADJ", "ADV"}

        # If the sentence is just conversational filler, force an immediate merge
        if sum(1 for t in doc2 if t.pos_ in content_pos) < 4:
            return -1.0

        # ── Standard Math ──
        # Batch: un solo encode() para los que falten (mismo patrón que
        # ProgressiveSegmenter.generate_embeddings / detect_topic_shift).
        emb1, emb2 = self._get_cached_embeddings([segment1, segment2])

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Handle zero division gracefully
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)

        cohesion = self.semantic_cohesion_score(segment1, segment2)

        return 0.5 * (1 - similarity) + 0.5 * (1 - cohesion)

    def _get_cached_embedding(self, text):
        """Compat de un solo texto. Para 2+ textos usa _get_cached_embeddings
        (batchea el encode() de los que falten en vez de uno por uno)."""
        return self._get_cached_embeddings([text])[0]

    def _get_cached_embeddings(self, texts):
        """Devuelve los embeddings de `texts` en el mismo orden, pidiendo al
        modelo en un solo batch únicamente los que falten en caché.

        Normalizado: la caché es compartida con ProgressiveSegmenter
        (generate_embeddings normaliza). Solo se usa para similitud coseno,
        que es invariante a la normalización -> mismo resultado que sin
        normalizar. Mismo texto -> mismo vector, así que el resultado es
        idéntico a pedirlos uno a uno.
        """
        # dict.fromkeys en vez de un set: preserva orden y deduplica (por si
        # segment1 == segment2 en el mismo llamado).
        missing = [
            t for t in dict.fromkeys(texts) if t not in self.embedding_cache
        ]
        if missing:
            embs = self.embedding_model.encode(
                missing,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=min(64, len(missing)),
                show_progress_bar=False,
            )
            for t, e in zip(missing, embs):
                self.embedding_cache[t] = e
        return [self.embedding_cache[t] for t in texts]

    def robust_sentence_split(self, text):
        doc = self.nlp(text)
        return [
            {"text": sent.text.strip(), "start": sent.start_char, "end": sent.end_char}
            for sent in doc.sents
        ]

    def segment_sentences(self, sentences, threshold=0.8):
        segments = [[sentences[0]]]
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]["text"]
            last_segment_text = " ".join([s["text"] for s in segments[-1]])
            if (
                self.compute_boundary_score(last_segment_text, current_sentence)
                < threshold
            ):
                segments[-1].append(sentences[i])
            else:
                segments.append([sentences[i]])
        return segments

    def segment_text(self, text, max_segments=10, threshold=0.5):
        sentences = self.robust_sentence_split(text)
        if not sentences:
            return [text]
        grouped = self.segment_sentences(sentences, threshold)

        # Precompute and cache boundary scores
        boundary_scores = [
            self.compute_boundary_score(
                " ".join(s["text"] for s in grouped[i]),
                " ".join(s["text"] for s in grouped[i + 1]),
            )
            for i in range(len(grouped) - 1)
        ]

        while len(grouped) > max_segments:
            merge_idx = int(np.argmin(boundary_scores))
            # Merge and update only the affected neighbors
            grouped[merge_idx] += grouped.pop(merge_idx + 1)
            boundary_scores.pop(merge_idx)
            if merge_idx < len(boundary_scores):
                boundary_scores[merge_idx] = (
                    self.compute_boundary_score(
                        " ".join(s["text"] for s in grouped[merge_idx]),
                        " ".join(s["text"] for s in grouped[merge_idx + 1]),
                    )
                    if merge_idx + 1 < len(grouped)
                    else float("inf")
                )

        return [" ".join(s["text"] for s in seg) for seg in grouped]


# ─────────────────────────────────────────────────────────────────────────────
# ProgressiveSegmenter  — with full debug instrumentation on coref pipeline
# ─────────────────────────────────────────────────────────────────────────────
class ProgressiveSegmenter:
    def __init__(
        self,
        model_name=DEFAULT_SEGMENTER_EMBEDDING_MODEL,
        stanza_lang="es",
        spacy_model=DEFAULT_SPACY_MODEL,
        nli_model=DEFAULT_NLI_MODEL,
        similarity_threshold=0.6,
        max_depth=3,
        window_size=3,
        hnsw_ef=200,
        hnsw_m=16,
        device=None,
        # ── new: control debug verbosity ────────────────────────────
        debug_coref=True,
        # Tope del caché de embeddings compartido (ver _BoundedEmbeddingCache).
        # None -> usa el default (KAG_EMBED_CACHE_SIZE o 20_000).
        embed_cache_size=None,
    ):
        self.similarity_threshold = similarity_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_depth = max_depth
        self.index = None
        self.embeddings = None
        self.window_size = window_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hnsw_ef = hnsw_ef
        self.hnsw_m = hnsw_m
        self.stanza_lang = stanza_lang
        self.stanza_use_gpu = device is not None
        self.debug_coref = debug_coref  # toggle for coref debug prints

        # Caché de embeddings compartida (texto -> vector), acotada con LRU:
        # esta instancia persiste entre documentos (ver _SEGMENTER_CACHE), así
        # que sin límite crecería para siempre. Solo optimiza rendimiento:
        # mismo texto -> mismo vector, nunca cambia el resultado.
        self._embed_cache = _BoundedEmbeddingCache(
            max_size=embed_cache_size or _default_embed_cache_size()
        )

        self.model = SentenceTransformer(model_name).to(self.device)
        self.classicseg = ClassicSegmenter(
            self.model,
            spacy_model=spacy_model,
            nli_model=nli_model,
            embed_cache=self._embed_cache,
        )
        self.nlp = spacy.load(spacy_model)
        if "conversational_sbd" not in self.nlp.pipe_names:
            # We insert it BEFORE the parser so the dependency parser respects these cuts
            self.nlp.add_pipe("conversational_sbd", before="parser")
        if "conversational_sbd" not in self.classicseg.nlp.pipe_names:
            self.classicseg.nlp.add_pipe("conversational_sbd", before="parser")
        # self.attention_detector = AttentionShiftDetector()  # now safe (device fixed)
        self.tfidf_vectorizer = TfidfVectorizer()
        self._stanza_pipeline = None
        logging.info(f"[Init] Device: {self.device}")

    # ── debug helper ──────────────────────────────────────────────────────────
    def _dprint(self, msg):
        """Print only when debug_coref is True."""
        if self.debug_coref:
            print(f"[COREF] {msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stanza lazy-load  FIX: was writing self.stanza_pipeline (no underscore)
    # which meant _stanza_pipeline was never set → pipeline rebuilt every call.
    # ─────────────────────────────────────────────────────────────────────────
    def get_stanza(self) -> Optional[stanza.Pipeline]:
        if self._stanza_pipeline is None:
            # Solo los idiomas con coref de Stanza tienen pipeline; el resto
            # desactiva la correferencia sin intentar descargar/cargar nada.
            if self.stanza_lang not in STANZA_COREF_LANGS:
               