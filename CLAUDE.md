# Release Team Intelligenz

## Purpose

Release Team Intelligenz is an AI-powered engineering assistant for QA teams.

Its goal is to generate standardized Test Plans by combining:

- Redmine tickets
- Workflow documentation
- Historical test cases
- Historical issue sheets
- Release notes

using Retrieval-Augmented Generation (RAG).

---

# Architecture

React (Frontend)

↓

FastAPI (Backend)

↓

Document Parser

↓

Embedding Service

↓

ChromaDB

↓

OpenAI API

↓

Generated Test Plan

---

# Design Principles

- Keep React focused on UI only.
- Never call OpenAI directly from React.
- Never connect React directly to ChromaDB.
- All AI requests go through FastAPI.
- Keep business logic inside the service layer.
- Repository layer handles database access.
- Services should remain modular and reusable.
- Prefer composition over inheritance.
- Write readable code rather than clever code.

---

# Coding Standards

- Small functions.
- Descriptive variable names.
- Type hints in Python.
- JSDoc for reusable JavaScript functions.
- Avoid duplicate logic.
- Follow SOLID principles where practical.

---

# Project Structure

frontend/

backend/

documents/

generated/

feedback/

vector_db/

---

# AI Workflow

User

↓

Redmine API

↓

Document Parsing

↓

Chunking

↓

Embedding Generation

↓

ChromaDB Retrieval

↓

Prompt Builder

↓

OpenAI

↓

Structured Test Plan

↓

Download Excel

---

# Future Roadmap

- Product Knowledge Assistant
- Automation Script Generator
- Regression Advisor
- Release Dashboard
- Customer Support Assistant

---

# Important Rules

When implementing a feature:

- Explain architectural decisions first.
- Avoid breaking existing APIs.
- Reuse existing services whenever possible.
- Keep components small.
- Prefer maintainability over shortcuts.
- If multiple approaches exist, explain trade-offs before coding.
## Test Isolation

All automated tests must run in a completely isolated environment and must never read from, modify, or delete the application's real runtime data under `backend/data/`. Tests should always use a temporary data root (e.g. `backend/data2/` or a pytest temporary directory) configured via dependency injection or test settings, and must clean up all test artifacts after execution, leaving the production `backend/data/` directory untouched.