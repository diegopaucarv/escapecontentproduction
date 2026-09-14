"""Tests de umbrales automáticos (src/kag/thresholds.py).

Funciones puras: Kneedle (codo), auto_cutoff y auto_margin. Sin DB, sin
torch/spacy.
"""

from src.kag.thresholds import auto_cutoff, auto_margin, knee_index

# ---------------------------------------------------------------------
# knee_index (Kneedle)
# ---------------------------------------------------------------------


def test_knee_index_power_law():
    # Curva convexa decreciente clara: el codo está en el índice 2.
    vals = [1.0, 0.5, 0.25, 0.125, 0.0625]
    assert knee_index(vals) == 2


def test_knee_index_head_tail():
    # Cabeza plana + cola: el codo está al inicio de la cola (índice 3).
    vals = [0.33, 0.33, 0.33, 0.05, 0.05, 0.007, 0.0001, 0.0001]
    assert knee_index(vals) == 3


def test_knee_index_flat_returns_none():
    assert knee_index([0.1, 0.09, 0.08, 0.07]) is None


def test_knee_index_linear_returns_none():
    assert knee_index([1.0, 0.8, 0.6, 0.4, 0.2]) is None


def test_knee_index_short_returns_none():
    assert knee_index([1.0, 0.5]) is None


def test_knee_index_all_equal_returns_none():
    assert knee_index([0.5, 0.5, 0.5, 0.5]) is None


# ---------------------------------------------------------------------
# auto_cutoff
# ---------------------------------------------------------------------


def test_auto_cutoff_uses_knee_score():
    vals = [0.9, 0.5, 0.3, 0.1, 0.05, 0.01]
    assert auto_cutoff(vals) == 0.1


def test_auto_cutoff_no_knee_returns_none():
    assert auto_cutoff([0.1, 0.09, 0.08, 0.07]) is None


def test_auto_cutoff_empty_returns_none():
    assert auto_cutoff([]) is None


# ---------------------------------------------------------------------
# auto_margin
# ---------------------------------------------------------------------


def test_auto_margin_median():
    gaps = [0.1, 0.3, 0.5, 0.7]
    assert auto_margin(gaps, percentile=0.5, floor=0.05) == 0.3


def test_auto_margin_floor_wins():
    # La mediana cae por debajo del piso → gana el piso.
    gaps = [0.02, 0.03, 0.04]
    assert auto_margin(gaps, percentile=0.5, floor=0.05) == 0.05


def test_auto_margin_few_gaps_uses_floor():
    assert auto_margin([0.1], percentile=0.5, floor=0.05) == 0.05


def test_auto_margin_empty_uses_floor():
    assert auto_margin([], percentile=0.5, floor=0.05) == 0.05
