# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local tool that scrapes job postings from LinkedIn, Naukri and Indeed via Playwright, then uses
OpenAI to read the user's resume and score each job against it. The backend (`src/app/`) is a
layered FastAPI application; the frontend (`frontend/index.html`) is a single self-contained
vanilla HTML/CSS/JS file with no build step. No data leaves the user's machine except resume text
and job descriptions sent to OpenAI.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; .venv\Scripts\Activate.ps1 for PowerShell
pip install -r requirements-dev.txt
pip install -e .                # one-time: makes `app` importable as a package
playwright install chromium

# Run the web UI
python -m app.main serve

# Run a one-off CLI search
python -m app.main search --resume my_resume.pdf --location Pune Remote --experience 2-4 --results linkedin=20 naukri=12

# Run the saved daily search once (normally invoked by Windows Task Scheduler / cron)
python -m app.main daily

# Tests (all offline: saved HTML fixtures in tests/fixtures/, fake OpenAI, fake job sites)
pytest
pytest tests/unit                      # fast, no browser
pytest tests/integration                # needs a real (headless) Chromium
pytest tests/unit/test_naukri_indeed.py -k naukri
```

There is no lint/format config (no ruff/black/mypy config files) — pytest runs via the
`[tool.pytest.ini_options]` block in `pyproject.toml` (sets `pythonpath = ["src"]`, so tests work
even without the editable install).

## Architecture

`src/app/` is layered: **api → services → repositories/models**, with `schemas/` and `config/` and
`database/` as supporting layers. Each layer only depends on the ones below it (`services` never
imports from `api`; `models` has no dependencies on anything else in the project).

- `models/` — plain dataclasses with no I/O: `Job`, `Profile`, `SearchOptions`, `RunStats`,
  `SearchResult`, `Schedule`. This is the bottom of the dependency graph.
- `config/` — `Settings`/`load_settings()` (`.env`-backed, command-line always wins) and static
  reference data (`LOCATION_CHOICES` etc.).
- `database/` — `Database`: a shared "fresh `sqlite3.connect()` per call + a lock" helper, factoring
  out a pattern duplicated between the two SQLite repositories.
- `repositories/` — the persistence boundary. `JobHistoryRepository` and `CacheRepository` own raw
  SQL and their own schemas (`history.sqlite3`: applied/hidden/seen jobs + run history, never
  auto-deleted; `cache.sqlite3`: resume readings, job descriptions, AI scores, content-hash keyed so
  reruns reuse prior AI calls — safe to delete, cleaned by `KEEP_DAYS`). `SessionRepository` and the
  schedule JSON functions persist what the web UI is currently showing and the saved daily-search
  config, respectively.
- `services/` — business logic and orchestration:
  - `services/scraping/` — `BaseScraper` ABC (shared retry/backoff: `goto()` retries on 429/503,
    `unreachable` trips after consecutive failures so one dead site can't block the others,
    `pause()` sleeps politely and aborts early on `stop_event`), `linkedin.py`/`naukri.py`/`indeed.py`
    implementations, and `browser.py`'s `SiteSession` — **one Playwright browser per site, each
    living on its own thread**, because Playwright's sync API isn't thread-safe across threads.
    `SCRAPERS` (in `scraping/__init__.py`) is the registry new boards get wired into.
  - `search_service.py` — the core flow: `run()` (resume → profile → search sites → dedupe/filter →
    fetch details → score → report), reporting progress through an `on_event(name, data)` callback
    (stages: `stage`, `profile`, `site`, `progress`, `funnel`) that both the API layer and CLI
    consume. `SearchService` is a thin class wrapper around `run()` for DI in `api/`.
  - `search_manager.py` — `SearchManager` runs one search at a time on a background `threading.Thread`
    so the HTTP API can poll status and support **Stop**; `SearchJob` is the in-memory/serializable
    status object.
  - `matching_service.py` — parsing/filtering/ranking helpers shared by scrapers and `search_service`
    (experience parsing, dedupe, keyword scoring, `SearchStopped` exception).
  - `llm_service.py` — `JobMatcherLLM`: resume → `Profile` extraction and parallel per-job scoring;
    tracks `TokenUsage` for cost display.
  - `cleanup_service.py`, `schedule_service.py`, `reporting/` (HTML + Excel/CSV output),
    `resume_service.py`, `location_service.py` — one concern each, named for what they do.
- `schemas/` — `ApiError` (raised by a route, converted to `{"error": message}` by a FastAPI
  exception handler registered in `main.py`) plus `int_in_range`/`str_list` validation helpers, and
  `serializers.py` (domain object → plain dict, what the frontend expects). Request bodies for the
  two complex endpoints (`/api/search`, `/api/schedule`) are read as raw dicts and validated with
  these helpers rather than strict Pydantic models, specifically so hand-written error messages
  reach the frontend unchanged.
- `api/` — one FastAPI router per resource (`resume_routes.py`, `search_routes.py`,
  `history_routes.py`, `schedule_routes.py`, `session_routes.py`, `config_routes.py`,
  `reports_routes.py`), each built by a `build_router(ctx: ApiContext)` function. `ApiContext`
  (`dependencies.py`) bundles the settings/repositories/services a request handler needs; it's
  constructed once in `main.py`'s `create_app()` — not via FastAPI's `Depends()` — so tests can keep
  injecting fakes through `create_app(settings=..., runner=..., analyzer=..., system_scheduler=...)`
  exactly as before.
- `main.py` — the single entry point. `create_app()` is the FastAPI app factory (also used directly
  by tests via `fastapi.testclient.TestClient`). `main(argv)` dispatches `python -m app.main
  {serve,search,daily}` to `_serve()` (uvicorn + browser auto-open), `_search()` (the CLI, mirrors
  the old `cli.py`), and `_daily()` (runs the saved schedule, mirrors the old `daily.py`).

### Adding a new job board

Create `src/app/services/scraping/<board>.py` with a class extending `BaseScraper` (implement
`search()`, and `fetch_description()` if the board's search results don't already include the full
description), then register it in `src/app/services/scraping/__init__.py`'s `SCRAPERS` dict.

### Testing approach

`tests/unit/` is fast and never touches a browser (pure logic, repositories against `tmp_path`
SQLite files, mocked OpenAI). `tests/integration/` either drives a real headless Chromium against
saved fixture HTML (`tests/fixtures/`) via the `browser`/`page` fixtures in `tests/conftest.py`, or
goes through `fastapi.testclient.TestClient` with a fake `runner` injected into `create_app()` (no
real browser, but exercises real FastAPI routing/validation). Note: calling
`search_service.run()`/`SearchService.run()` directly **always** launches a real Chromium process via
`SiteSession`, even when the scraper class itself is fully faked — that's why those tests live in
`integration/` rather than `unit/`. When changing scraper selectors, update against the fixture files
rather than live sites.
