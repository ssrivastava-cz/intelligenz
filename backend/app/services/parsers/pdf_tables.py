"""Layout-aware table recovery for `PdfParser`.

Kept in its own module so `pdf_parser.py` stays small (see CLAUDE.md).
Nothing here touches embeddings/ChromaDB/RAG — it only turns the *ruled*
tables `pdfplumber` can see into the same "Table N / Header: value"
section shape `WordParser` already produces for DOCX tables, so a
retrieved chunk keeps its column relationships (`Plan ID: PLAN-A`) instead
of the ambiguous flat form pypdf gives (`Plan ID PLAN-A`).

Safety-first, by design:

* Only bordered tables are considered — detection uses `pdfplumber`'s
  "lines" strategy, which keys off actual ruling lines / rectangle
  edges. Whitespace-aligned text is never inferred to be a table.
* Every candidate must additionally pass `is_reliable_table` (a real
  header row, enough filled cells, more than one populated column, …).
* Anything that fails is left completely untouched — `PdfParser`'s
  existing pypdf text path then handles that region exactly as before.

A false positive here would be worse than the status quo, so the gate
errs towards rejecting.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.parsed_document import DocumentSection

# Border/rule based only. No `vertical_strategy="text"` — that is the
# path that turns aligned prose into fake columns.
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "edge_min_length": 10,
}

# A bbox edge within this many points of the page's top / bottom counts
# as "touching" it — used only to decide whether a borderless-looking
# fragment at the top of a page is the continuation of the table that
# ran to the bottom of the previous page.
_EDGE_MARGIN = 72  # 1 inch

_MAX_HEADER_CELL_LENGTH = 60
_MAX_LONELY_DATA_ROW_RATIO = 0.4
_MIN_FILL_RATIO = 0.5


def clean_rows(raw_rows: list[list[str | None]]) -> list[list[str]]:
    """Normalises a `pdfplumber` table extract: collapse internal
    whitespace in each cell, drop fully-empty rows, pad rows to equal
    width, then drop any column that is empty in every row.
    """
    rows = [[_clean_cell(cell) for cell in row] for row in raw_rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    kept_columns = [index for index in range(width) if any(row[index] for row in rows)]
    return [[row[index] for index in kept_columns] for row in rows]


def is_reliable_table(rows: list[list[str]]) -> tuple[bool, str]:
    """Whether `rows` (already `clean_rows`-normalised) is safe to render
    as a structured table. Returns `(ok, reason)` — `reason` names the
    failed check so a caller can log why a candidate was rejected.

    The thresholds were tuned against the real Source of Truth PDFs: the
    genuine bordered tables (glossaries, BR-### requirement grids,
    TC-### test-case grids) all pass; the decorative "PURPOSE" /
    "ASSUMPTION" callout boxes and the whitespace-gridded checklist all
    fail.
    """
    if len(rows) < 2:
        return False, "fewer than 2 non-empty rows"

    column_count = len(rows[0])
    if column_count < 2:
        return False, "fewer than 2 non-empty columns"

    populated_columns = sum(1 for index in range(column_count) if any(row[index] for row in rows))
    if populated_columns < 2:
        return False, "only one column carries content (callout/aside box)"

    total_cells = len(rows) * column_count
    filled_cells = sum(1 for row in rows for cell in row if cell)
    if filled_cells / total_cells < _MIN_FILL_RATIO:
        return False, f"low fill ratio ({filled_cells}/{total_cells})"

    header = rows[0]
    if not all(header):
        return False, "header row has a blank cell"
    if any(len(cell) > _MAX_HEADER_CELL_LENGTH for cell in header):
        return False, "header cell too long (looks like wrapped prose)"

    data_rows = rows[1:]
    lonely = sum(1 for row in data_rows if sum(1 for cell in row if cell) <= 1)
    if lonely / len(data_rows) > _MAX_LONELY_DATA_ROW_RATIO:
        return False, "too many data rows with a single populated cell (prose)"

    return True, "ok"


def column_count(rows: list[list[str]]) -> int:
    return len(rows[0]) if rows else 0


def starts_near_top(bbox: tuple[float, float, float, float], page_height: float) -> bool:
    top = bbox[1]
    return top <= _EDGE_MARGIN


def ends_near_bottom(bbox: tuple[float, float, float, float], page_height: float) -> bool:
    bottom = bbox[3]
    return bottom >= page_height - _EDGE_MARGIN


@dataclass(frozen=True)
class TableCarry:
    """The last accepted table's shape, carried to the next page so a
    top-of-page fragment can be tested as its continuation."""

    number: int
    header: list[str]
    column_count: int
    start_page: int
    ended_at_bottom: bool


def stitch_continuation(
    rows: list[list[str]],
    bbox: tuple[float, float, float, float],
    page_height: float,
    carry: TableCarry,
) -> list[list[str]] | None:
    """If `rows` is the top-of-page fragment of the table described by
    `carry` (which ran off the bottom of the previous page), return just
    its data rows — ready to render under `carry.header`. Otherwise
    `None`.

    Two shapes are accepted, both demanding the same column count and
    physical adjacency to the page break:

    * the previous header is *repeated* as `rows[0]` — drop it, keep the
      rest;
    * there is *no* header, just data, optionally preceded by a wrapped
      cell fragment bleeding down from the previous page's last row —
      trim leading rows that are not fully populated, keep the rest.

    In the second shape the fragment must NOT itself look like a
    self-contained table (`is_reliable_table`), so a genuinely new table
    that merely happens to start at the top of a page with the same
    column count is left to be recognised on its own. Every retained row
    must have all cells populated, so a partial row is never promoted to
    a full one.
    """
    if not carry.ended_at_bottom:
        return None
    if not starts_near_top(bbox, page_height):
        return None
    if column_count(rows) != carry.column_count:
        return None

    width = carry.column_count

    if rows[0] == carry.header:
        body = rows[1:]
    else:
        if is_reliable_table(rows)[0]:
            return None
        body = list(rows)
        while body and sum(1 for cell in body[0] if cell) < width:
            body.pop(0)

    if not body:
        return None
    if any(sum(1 for cell in row if cell) < width for row in body):
        return None
    return body


def render_table_section(
    table_number: int,
    header: list[str],
    data_rows: list[list[str]],
    page_number: int | None,
    *,
    continued: bool = False,
) -> DocumentSection:
    """One `DocumentSection` for a table — body is one self-contained
    "Header: value" block per data row, never a single concatenated
    string. This mirrors `WordParser._render_table` / `CsvParser` so a
    chunk boundary that lands between two rows of a large table still
    leaves each fragment naming its table and its columns.
    """
    label = f"Table {table_number}" + (" (continued)" if continued else "")

    if not data_rows:
        body = f"{label}\n" + " | ".join(header)
        return DocumentSection(heading=label, content=body, page_number=page_number)

    blocks: list[str] = []
    for row in data_rows:
        pairs = [f"{_column_label(header, index)}: {_cell(row, index)}" for index in range(len(header))]
        blocks.append(f"{label}\n" + "\n".join(pairs))
    return DocumentSection(heading=label, content="\n\n".join(blocks), page_number=page_number)


def _clean_cell(cell: str | None) -> str:
    return " ".join((cell or "").split())


def _column_label(header: list[str], index: int) -> str:
    if index < len(header) and header[index]:
        return header[index]
    return f"Column {index + 1}"


def _cell(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""
