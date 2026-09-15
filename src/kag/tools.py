"""
Herramientas de ingesta documental para el rediseño KAG.

Tres herramientas independientes, sin dependencias pesadas a nivel de módulo
(httpx se importa perezosamente dentro de los métodos que lo usan):

  - MultibookFinderTool: detecta libros/papers apilados en un único .md grande
    (anclas ISBN + separadores pesados, búsqueda hacia atrás de fronteras
    reales) y devuelve los límites físicos (line_start/line_end) de cada
    documento.
  - LibraryOfCongressAPITool: cliente de la API de Linked Data de la Biblioteca
    del Congreso (id.loc.gov) para encabezamientos de materia (LCSH) y códigos
    de clasificación (LCC) — ficha documental ISO 25964.
  - MarkdownImageExtractorTool: rastrea referencias de imágenes (sintaxis
    Markdown y tags HTML <img>) dentro del rango físico de un documento y
    resuelve la ruta absoluta en disco.

Todas son clases ligeras: solo `re` y `pathlib` a nivel de módulo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar


class MultibookFinderTool:
    """
    Herramienta de inspección de límites físicos para segmentar
    libros y papers apilados en un único archivo Markdown.
    """

    KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "pt": [
            "Introdução",
            "Colaboradores",
            "Índice",
            "Conteúdo",
            "Sumário",
            "Bibliografia",
            "Referências",
            "Prefácio",
            "ISBN",
        ],
        "fr": [
            "Introduction",
            "Contributeurs",
            "Index",
            "Contenu",
            "Sommaire",
            "Bibliographie",
            "Références",
            "Préface",
            "ISBN",
        ],
        "en": [
            "Introduction",
            "Contributors",
            "Index",
            "Contents",
            "Bibliography",
            "References",
            "Preface",
            "ISBN",
        ],
        "de": [
            "Einleitung",
            "Einführung",
            "Mitwirkende",
            "Index",
            "Register",
            "Inhalt",
            "Inhaltsverzeichnis",
            "Bibliografie",
            "Referenzen",
            "Vorwort",
            "ISBN",
        ],
        "es": [
            "Introducción",
            "Colaboradores",
            "Índice",
            "Contenido",
            "Bibliografía",
            "Referencias",
            "Prefacio",
            "ISBN",
        ],
    }

    ISBN_REGEX = re.compile(
        r"(?:ISBN(?:-1[03])?:?\s*)(?=[0-9X]{10}|(?=(?:[0-9]+[-\s]){3,4})[0-9-\sX]{13,17})",
        re.IGNORECASE,
    )
    TITLE_HEADER_REGEX = re.compile(r"^#\s+(.+)$", re.MULTILINE)
    SEPARATOR_REGEX = re.compile(r"^\s*[-=_]{3,}\s*$")

    def __init__(self, context_window_lines: int = 300):
        self.context_window = context_window_lines
        self.all_keywords = {
            kw.lower() for lang in self.KEYWORDS.values() for kw in lang
        }

    def execute(self, file_path: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines == 0:
            return []

        # 1. Localizar anclas: coincidencias de ISBN y separadores pesados
        marker_indices = [0]
        for idx, line in enumerate(lines):
            line_str = line.strip()
            if self.ISBN_REGEX.search(line_str) or self.SEPARATOR_REGEX.match(line_str):
                marker_indices.append(idx)

        # 2. Búsqueda hacia atrás de fronteras reales (separadores o bloques vacíos)
        split_candidates = [0]
        for idx in marker_indices[1:]:
            found_sep = False
            scan_floor = max(0, idx - self.context_window)

            for j in range(idx, scan_floor, -1):
                if self.SEPARATOR_REGEX.match(lines[j]):
                    split_candidates.append(j)
                    found_sep = True
                    break

            if not found_sep:
                for j in range(idx, scan_floor, -1):
                    # 4 o más saltos de línea consecutivos indican corte editorial
                    if j >= 3 and all(
                        lines[j - offset].strip() == "" for offset in range(4)
                    ):
                        split_candidates.append(j)
                        break

        split_candidates = sorted(set(split_candidates))
        if split_candidates[-1] != total_lines:
            split_candidates.append(total_lines)

        # 3. Diferenciación y extracción de documentos
        documents = []
        for i in range(len(split_candidates) - 1):
            start = split_candidates[i]
            end = split_candidates[i + 1]
            if (end - start) < 50:  # Ignorar fragmentos residuales
                continue

            doc_slice = lines[start : min(start + 100, end)]
            doc_text = "".join(doc_slice)

            # Inferencia del título
            title_match = self.TITLE_HEADER_REGEX.search(doc_text)
            title = (
                title_match.group(1).strip() if title_match else f"Documento_{i + 1}"
            )

            # Conteo de palabras clave presentes en la cabecera
            kw_hits = sum(1 for kw in self.all_keywords if kw in doc_text.lower())

            doc_id = f"{path.stem}_doc_{i + 1:03d}"
            documents.append(
                {
                    "source_file": str(path),
                    "document_id": doc_id,
                    "title": title,
                    "line_start": start + 1,
                    "line_end": end,
                    "total_lines": end - start,
                    "structural_keyword_hits": kw_hits,
                }
            )

        return documents


class LibraryOfCongressAPITool:
    """
    Cliente para la API de Linked Data de la Biblioteca del Congreso (id.loc.gov).
    Extrae encabezamientos de materia (LCSH) y códigos de clasificación (LCC).

    Usa httpx (no requests — no está en requirements.txt). La API devuelve
    JSON con la forma [query, [labels], [uris], [ids]]; se toma el primer
    candidato de cada lista. Cualquier error de red/HTTP degrada a None (la
    ficha documental se completa sin LCSH/LCC, no revienta la ingesta).
    """

    SUGGEST_URL = "https://id.loc.gov/authorities/subjects/suggest2/"
    CLASS_SEARCH_URL = "https://id.loc.gov/authorities/classification/suggest2/"

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def query_subject_heading(self, term: str) -> dict[str, str] | None:
        """Consulta LCSH para un término y devuelve {preferred_label, uri}.

        None si la API no responde, devuelve un status != 200 o el JSON no
        tiene la forma esperada.
        """
        import httpx

        params = {"q": term, "count": 1}
        try:
            r = httpx.get(self.SUGGEST_URL, params=params, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                if len(data) >= 4 and data[1]:
                    return {
                        "preferred_label": data[1][0],
                        "uri": data[3][0],
                    }
        except (httpx.HTTPError, httpx.TimeoutException):
            pass
        return None

    def query_classification_code(self, label: str) -> dict[str, str] | None:
        """Consulta LCC para una etiqueta y devuelve {lcc_call_number, lcc_uri}.

        None si la API no responde, devuelve un status != 200 o el JSON no
        tiene la forma esperada.
        """
        import httpx

        params = {"q": label, "count": 1}
        try:
            r = httpx.get(self.CLASS_SEARCH_URL, params=params, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                if len(data) >= 4 and data[1]:
                    return {
                        "lcc_call_number": data[1][0],
                        "lcc_uri": data[3][0],
                    }
        except (httpx.HTTPError, httpx.TimeoutException):
            pass
        return None


class MarkdownImageExtractorTool:
    """
    Rastrea referencias de imágenes en el texto Markdown de un documento,
    asocia el número de línea física y resuelve la ruta absoluta en disco.
    """

    # Soporta sintaxis estándar ![alt](path "title") y etiquetas HTML <img ...>
    MD_IMG_REGEX = re.compile(r'!\[(.*?)\]\((.+?)(?:\s+"(.*?)")?\)')
    HTML_IMG_REGEX = re.compile(
        r'<img[^>]+src=["\']([^"\']+)["\'](?:[^>]*alt=["\']([^"\']*)["\'])?[^>]*>',
        re.IGNORECASE,
    )

    def extract_images_from_document(
        self,
        source_file: str,
        document_id: str,
        line_start: int,
        line_end: int,
    ) -> list[dict[str, Any]]:
        base_path = Path(source_file).parent
        images_found = []

        with open(source_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Escanear el rango delimitado para este documento
        doc_slice = lines[line_start - 1 : line_end]

        for offset, raw_line in enumerate(doc_slice):
            current_line_num = line_start + offset

            # 1. Búsqueda de tags Markdown
            for match in self.MD_IMG_REGEX.finditer(raw_line):
                alt_text, raw_path, title = match.groups()
                clean_path = raw_path.strip().split()[0]  # Limpieza de parámetros
                resolved_path = (base_path / clean_path).resolve()

                images_found.append(
                    {
                        "document_id": document_id,
                        "anchor_line": current_line_num,
                        "markdown_tag": match.group(0),
                        "caption": alt_text.strip()
                        if alt_text
                        else (title.strip() if title else ""),
                        "file_path": str(resolved_path),
                        "exists_on_disk": resolved_path.is_file(),
                    }
                )

            # 2. Búsqueda de tags HTML <img>
            for match in self.HTML_IMG_REGEX.finditer(raw_line):
                src, alt = match.groups()
                resolved_path = (base_path / src.strip()).resolve()

                images_found.append(
                    {
                        "document_id": document_id,
                        "anchor_line": current_line_num,
                        "markdown_tag": match.group(0),
                        "caption": alt.strip() if alt else "",
                        "file_path": str(resolved_path),
                        "exists_on_disk": resolved_path.is_file(),
                    }
                )

        return images_found
