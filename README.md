# Release Team Intelligenz

AI-assisted release intelligence platform.

**This is a development-first setup.** Everything runs locally with plain
`npm run dev` / `uvicorn --reload` — no Docker, no Kubernetes, no CI/CD.
Docker support will be added later, once Version 1 is complete.

## Architecture

The frontend **never** communicates directly with OpenAI, MongoDB, or
ChromaDB. All requests flow through the FastAPI backend, which is the
sole owner of data access and AI calls — that integration hasn't been
built yet, though: the backend currently returns realistic mock data
for every endpoint so the frontend can be built and demoed against a
stable contract before real AI/DB logic lands.

```
┌───────────┐      HTTP/JSON       ┌────────────────┐
│  Frontend │ ───────────────────▶ │  FastAPI Backend │
│  (React)  │ ◀─────────────────── │   (mock data)   │
└───────────┘                      └───────┬─────────┘
                                            │  planned, not yet wired up
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
               ┌───────────┐         ┌────────────┐         ┌────────────┐
               │  MongoDB  │         │  ChromaDB  │         │ OpenAI API │
               └───────────┘         └────────────┘         └────────────┘
```

## Tech Stack

| Layer          | Technology              |
|----------------|--------------------------|
| Frontend       | React + Vite (JavaScript) |
| Backend        | FastAPI (Python)         |
| Database       | MongoDB (local) — planned, not yet integrated |
| Vector Store   | ChromaDB (local, embedded `PersistentClient`) — planned, not yet integrated |
| AI             | OpenAI API — planned, not yet integrated |

## Backend Structure

```
backend/app/
├── routers/     # FastAPI route handlers — one module per resource
├── services/    # Business logic; each router depends on a service, not the other way around
├── models/      # Internal domain entities (snake_case, framework-agnostic)
├── schemas/     # API request/response DTOs (camelCase over the wire)
├── config/      # Environment-driven settings
├── core/        # Cross-cutting concerns: DI providers, exceptions, error handling, logging
├── utils/       # Small stateless helpers (ids, datetime, CSV export, mock fixtures)
└── main.py      # App entrypoint
```

Routers depend on services via FastAPI's `Depends` (see
`app/core/dependencies.py`) rather than instantiating them directly —
swapping a mock service for a real one later (e.g. wiring
`TestPlanService` to OpenAI/ChromaDB) won't require touching any
router. `schemas/` are kept separate from `models/` so the API
contract (camelCase, matching the frontend) can evolve independently
of internal representation.

Every endpoint currently returns mock data — see `app/utils/mock_data.py`
for the fixtures and `app/services/test_plan_service.py` for how
`/generate-test-plan` + `/status` simulate a processing pipeline
without a real background worker.

### Document Ingestion

One exception to "everything is mocked": `UploadService`
(`app/services/upload_service.py`) does real, validated file storage —
no parsing, embeddings, or AI. It handles both document sources:

| Source | Endpoint | Storage |
|---|---|---|
| Source of Truth (approved knowledge base) | `POST /source-of-truth/{feature}/documents` | `<STORAGE_ROOT>/source_of_truth/<feature>/` |
| User upload (attached while generating a test plan) | `POST /upload-documents` | `<STORAGE_ROOT>/uploads/<sessionId>/` |

Documents are uploaded *before* a generation exists, so uploads are
grouped by an **upload session**, not a generation:

1. `POST /upload-session` → `{ sessionId }`
2. `POST /upload-documents` (`sessionId`, `feature`, `files`) — any number of times
3. `POST /generate-test-plan` (`sessionId`, `feature`, `redmineId`, `description`) —
   pulls in every document uploaded to that session, then creates the
   `generationId`

`generationId` only exists from step 3 onward, and is what
`/status`, `/history`, `/download`, and `/feedback` continue to use.
A generation can also be created with no session at all (no documents
attached) — sessions are optional, not required.

Both upload endpoints validate file extension (PDF, DOCX, XLSX, CSV,
TXT, Markdown), max size (`MAX_UPLOAD_SIZE_MB`, default 20), and
duplicate filenames (within a request and against what's already
stored), and sanitize `feature`/`sessionId`/filename inputs against
path traversal. `STORAGE_ROOT` defaults to `./data` (gitignored). Each
stored `Document` records a `storage_path` — the contract `ParserService`
(below) uses to read the file; `UploadService.get_document` /
`list_source_of_truth_documents` / `list_session_documents` are that
lookup surface.

**Backward compatibility**: `/upload-documents` still accepts the
pre-session `generationId` field name as an alias for `sessionId`
(lazily registering it as a session if `/upload-session` was never
called), and `/generate-test-plan` still accepts the pre-rename
`ticketId` field name as an alias for `redmineId`.

### Document Parsing

`app/services/parsers/` turns a stored file back into structured text
— still no OpenAI, embeddings, ChromaDB, or RAG; just extraction.
Every parser implements the same interface (`DocumentParser.parse`,
`app/services/parsers/base.py`) and returns the same shape
(`ParsedDocument`, `app/models/parsed_document.py`):

```
document_id, title, sections[{heading, content, page_number}], content,
metadata{feature, document_type, document_source, source_filename, page_number,
         parser_name, parser_version}
```

`parser_name`/`parser_version` identify which parser produced a given
`ParsedDocument` (e.g. `"PdfParser"` / `"1.0"`) — every parser
populates both; all current parsers are version `"1.0"`.

| Format | Parser | Sectioning |
|---|---|---|
| PDF | `PdfParser` | one section per detected heading (see below); falls back to one section per page if none are found |
| DOCX | `WordParser` | one section per `Heading N` style |
| XLSX | `ExcelParser` | one section per sheet |
| CSV | `CsvParser` | **one section per row** — not one section for the whole file |
| Markdown | `MarkdownParser` | one section per `#`/`##`/... heading |
| TXT | `TxtParser` | one section per detected heading, or a single section if none are found |

Parsers expose logical structure, not just extracted text — this is
what lets `ChunkingEngine` (below) stay simple: a section is meant to
already be the right *unit* to turn into one chunk.

- **`CsvParser`** parses one `DocumentSection` per data row (via
  `csv.DictReader`), with every column rendered into the section body
  regardless of name — so a CSV of historical test cases or issues
  naturally becomes one section per test case / issue. The one
  schema-aware bit: if the row has a column ending in "id" (e.g. "Test
  Case ID", "Issue ID") and/or a "Title" column, the section heading
  combines them ("TC-001 - Book appointment"); otherwise the section
  is unheaded.
- **`TxtParser`** and **`PdfParser`** share `looks_like_heading()`
  (`app/utils/text.py`) — a heuristic for formats with no native
  heading markup: a short line that doesn't trail off mid-sentence and
  is either ALL CAPS, ends with a colon, or is Title Case ("Role
  Permissions"). `TxtParser` applies it to blank-line-delimited
  blocks; `PdfParser` applies it within each page's extracted text,
  and a section can span a page boundary (keeping the page it started
  on). If no line in the whole PDF matches, `PdfParser` falls back to
  its original one-section-per-page behavior — a heuristic that finds
  nothing shouldn't produce worse structure than not trying at all.
- `TxtParser` reads UTF-8 and falls back to `latin-1` (which can't
  fail — every byte maps to a code point) if a file isn't valid
  UTF-8, so it never raises a decode error the way
  `CsvParser`/`MarkdownParser` do.

`ParserService.parse_document(document_id)` looks the `Document` up
via `UploadService`, reads its `storage_path`, and dispatches to the
right parser by `document_type` — it's source-agnostic by
construction, so it works identically for Source of Truth and
user-uploaded documents (only `Document.source` differs, and no
parser branches on it). Every supported upload extension now has a
parser. `ParserService.parse_content(document, content)` does the same
dispatch directly from a `Document` + bytes, for callers (like
`SourceOfTruthIndexer` below) that don't go through `UploadService` at
all — `parse_document` is just `parse_content` plus the lookup/read.

### Source of Truth Indexing

`SourceOfTruthIndexer` (`app/services/source_of_truth_indexer.py`)
scans a fixed-shape folder tree — separate from the flat
`source_of_truth/<feature>/` layout `UploadService` writes to — and
turns it into `Document`s the same `ParserService` parses:

```
source_of_truth/<feature>/workflows/...    → category WORKFLOW
source_of_truth/<feature>/TestCases/...    → category TEST_CASE
source_of_truth/<feature>/IssueSheets/...  → category ISSUE
```

**`category` is a new, separate field from `document_type`.** The
folder gives you *what kind of Source of Truth content this is*
(`DocumentCategory`: WORKFLOW/TEST_CASE/ISSUE); the file *format*
(`DocumentType`: PDF/DOCX/.../for parser dispatch) still comes from
the extension, exactly like `UploadService` derives it — a `.csv`
under `workflows/` is category `WORKFLOW`, format `CSV`, and gets
parsed by `CsvParser` regardless of which folder it was found in.
Conflating the two would have broken `ParserService`'s dispatch table
(which is keyed on file format), so they're independent axes on
`Document` rather than one field repurposed.

Two read-only, GET-only endpoints:

| Endpoint | Returns |
|---|---|
| `GET /source-of-truth` | `[{feature, documents, workflows, testCases, issues}]` — counts per feature |
| `GET /source-of-truth/{feature}/parsed` | every document for a feature, parsed — `documentName`, `documentType` (the category), the full `parsedDocument`, `sectionCount`, `sectionHeadings` |

`/parsed` is for inspecting parser output before chunking — it calls
`ParserService` and returns exactly what it produces, nothing more:
no chunks, no embeddings. Files with an unrecognized extension are
silently skipped during a scan rather than failing it; an unknown
`{feature}` on `/parsed` is a 404.

### Chunking Engine

`ChunkingEngine` (`app/services/chunking_engine.py`) turns a
`ParsedDocument` into `Chunk`s — still no embeddings, no ChromaDB, just
text splitting. Not wired to any endpoint yet (same reasoning as
`/parsed` above: chunking is a deliberately separate, later stage).

**The engine is format- and category-agnostic — on purpose.** It does
not branch on `artifact_type`, or on anything else about the document.
It does exactly one thing: convert every `DocumentSection` into one
chunk, splitting only a section that exceeds `chunk_size` words into
overlapping windows (`chunk_overlap` words of overlap between
consecutive windows). That's the whole engine.

"One chunk per Test Case" and "one chunk per Issue" are consequences
of the *parser*, not the engine: `CsvParser` produces one section per
CSV row, so a normally-sized test case or issue section just passes
through as a single chunk without the engine needing to know what a
"test case" is. Earlier this engine had an `artifact_type`-keyed
branch that special-cased TEST_CASE/ISSUE into unsplit single chunks
— that branch is gone; the responsibility moved to `CsvParser`
exposing the right logical unit in the first place. `artifact_type`
is still a required argument to `chunk_document(parsed_document,
artifact_type)`, since it's still part of the `Chunk` schema, but it's
now pure metadata the caller supplies, not a signal the engine reads —
`ParsedDocument` doesn't carry Source of Truth classification itself
(that's `Document.category`, which parsers never see), so there's no
way for the engine to infer it even if it wanted to.

`artifact_type` reuses `DocumentCategory` (extended with
`RELEASE_NOTES` and `REQUIREMENT`, since chunking needs to classify
more kinds of content than the Source of Truth folder structure
currently produces) rather than introducing a second, overlapping
enum.

`chunk_size`/`chunk_overlap` are configurable (constructor args on
`ChunkingEngine`, or `CHUNK_SIZE`/`CHUNK_OVERLAP` env vars via
`ChunkingEngineDep`) and validated at construction (`chunk_overlap`
must be smaller than `chunk_size`). Every `Chunk` carries `chunk_id`,
`document_id`, `feature`, `artifact_type`, `document_source`,
`source_filename`, `page_number` (inherited from the section it came
from), `chunk_number` (sequential across the whole document), and
`chunk_text`.

## Frontend Structure

```
frontend/src/
├── components/    # Reusable, generic UI atoms (Button, Card, Badge, ...)
├── layout/        # App shell: Sidebar, TopNavbar, AppLayout
├── pages/         # Route-level views, each with its own components/ subfolder
├── hooks/         # Custom React hooks
├── config/        # Static UI config (nav items, analysis options, upload rules)
├── mocks/         # Mock JSON data the UI runs on
├── services/      # Mock async pipelines (e.g. simulated test-plan generation)
├── utils/         # Pure helper functions
├── styles/        # CSS variables + globals (no framework)
└── api/           # HTTP client, not yet wired to any page (backend integration is next)
```

## Getting Started

Everything runs locally — no containers required.

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

No database or vector store setup is needed yet — every endpoint returns
mock data, so the backend runs standalone. Interactive API docs are at
`http://localhost:8000/docs` once it's running.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

## Status

- **Frontend**: AI Test Plan Generator, Home, and Usage Dashboard pages are built and fully functional against local mock JSON — no backend calls yet.
- **Backend**: all planned endpoints exist and return realistic mock JSON (see `app/utils/mock_data.py`) — no OpenAI, ChromaDB, MongoDB, or Redmine integration yet.
- Frontend and backend are not wired together yet.

## Roadmap

Docker (and, later, Kubernetes/CI-CD) will be introduced after Version 1 is
functionally complete. Until then, the project intentionally stays
dependency-light and runs entirely from local dev servers.
