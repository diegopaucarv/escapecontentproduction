"""
Umbrales automáticos (Tier 1) — técnicas estadísticas estándar.

  - knee_index: algoritmo Kneedle (Satopää et al., 2011) para detectar el
    codo de una curva decreciente (rank, score). Es el método estándar de
    detección de codos; reemplaza umbrales absolutos que no escalan con el
    tamaño del grafo ni la distribución de grados.
  - auto_cutoff: score de corte = score en el codo (Kneedle). Devuelve None
    si no hay codo claro — el caller usa su fallback relativo.
  - auto_margin: margen = mediana (percentil robusto) de los gaps
    observados, con piso. Para desambiguación y canonicalización: si dos
    candidatos están igual de cerca, el margen decide si se resuelve o se
    queda ambiguo.

Funciones puras, sin dependencias pesadas — compartidas por el querying
(src/kag_query.py) y la ingesta (src/kag/entities.py).
"""

from __future__ import annotations


def knee_index(
    values_desc,
    sensitivity: float = 1.0,
    min_points: int = 3,
    min_curvature: float = 0.01,
):
    """Índice del codo (Kneedle) en una curva decreciente (rank, score).

    `values_desc` es la lista de scores ordenada DESCENDENTE. Normaliza
    rank→[0,1] y score→[0,1], calcula la curva diferencia D = y_line - y
    (positiva en el codo de una curva convexa decreciente) y devuelve el
    primer máximo local de D que supera D_max - S·mean(D). Devuelve None
    si la curva es demasiado corta, plana o casi lineal (sin codo claro →
    el caller usa su fallback relativo).
    """
    n = len(values_desc)
    if n < min_points:
        return None
    vmin, vmax = values_desc[-1], values_desc[0]
    if vmax - vmin < 1e-12:
        return None
    x = [i / (n - 1) for i in range(n)]
    y = [(v - vmin) / (vmax - vmin) for v in values_desc]
    y_line = [1.0 - xi for xi in x]
    d = [yl - yi for yi, yl in zip(y, y_line)]
    d_max = max(d)
    if d_max < min_curvature:
        return None  # curva casi lineal: sin codo claro
    threshold = d_max - sensitivity * (sum(d) / len(d))
    for i in range(1, n - 1):
        if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] >= threshold:
            return i
    return None


def auto_cutoff(values_desc):
    """Score de corte automático: score en el codo (Kneedle).

    Devuelve None si no hay codo claro (curva plana/corta) — el caller usa
    su fallback relativo. Si hay codo, el dato manda: el corte es el score
    en el codo.
    """
    if not values_desc:
        return None
    knee = knee_index(values_desc)
    if knee is None:
        return None
    return values_desc[knee]


def auto_margin(gaps, percentile: float = 0.5, floor: float = 0.05):
    """Margen automático: percentil (default mediana) de los gaps, con piso.

    Con <2 gaps no hay distribución → devuelve el piso. La mediana es la
    medida de tendencia central más robusta: la mitad de las menciones con
    señal se resuelven, la otra mitad queda ambigua.
    """
    if len(gaps) < 2:
        return floor
    g = sorted(gaps)
    idx = min(len(g) - 1, int(percentile * (len(g) - 1)))
    return max(g[idx], floor)
