"""Single entry point: `python -m app.main <serve|search|daily>`.

  serve   the local web UI (was: python -m job_search.web)
  search  one-off command-line search (was: python -m job_search)
  daily   run the saved daily search (was: python -m job_search.daily)

It only listens on your own computer (127.0.0.1), never on the network, in serve mode.
"""

import argparse
import logging
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAIError

from app.api.dependencies import ApiContext
from app.api import config_routes, history_routes, reports_routes, resume_routes, schedule_routes, search_routes, session_routes
from app.config import Settings, load_settings
from app.models import Profile, SearchOptions
from app.repositories import CacheRepository, JobHistoryRepository, SessionRepository, load_schedule, resume_key, save_state
from app.repositories.schedule_repository import LOG_FILE
from app.schemas.common import ApiError
from app.schemas.serializers import suggest_experience
from app.services.cleanup_service import cleanup
from app.services.llm_service import JobMatcherLLM
from app.services.matching_service import parse_experience_option
from app.services.resume_service import read_resume_text
from app.services.schedule_service import SystemScheduler, options_from_dict
from app.services.scraping import SCRAPERS
from app.services.search_manager import SearchManager
from app.services.search_service import run

log = logging.getLogger(__name__)

# src/app/main.py -> app -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ---------- FastAPI app ----------

def default_analyzer(settings: Settings):
    def analyze(resume_path: Path) -> Profile:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
        # The same resume file (by content) is only sent to the AI once.
        cache = CacheRepository(settings.cache_path)
        key = resume_key(resume_path, settings.openai_model)
        profile = cache.get_profile(key)
        if profile is None:
            text = read_resume_text(resume_path)
            profile = JobMatcherLLM(settings.openai_api_key, settings.openai_model).extract_profile(text)
            cache.save_profile(key, profile)
        return profile

    return analyze


def create_app(
    settings: Settings | None = None,
    output_dir: Path = Path("output"),
    upload_dir: Path = Path("uploads"),
    runner=run,
    analyzer=None,
    system_scheduler: SystemScheduler | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    analyzer = analyzer or default_analyzer(settings)
    data_dir = settings.history_path.parent
    scheduler = system_scheduler or SystemScheduler(data_dir)
    try:
        cleanup(settings.keep_days, data_dir, upload_dir, output_dir, settings.cache_path)
    except Exception:  # housekeeping must never stop the app from starting
        log.exception("Cleanup failed")
    session = SessionRepository(data_dir)
    manager = SearchManager(settings, output_dir, runner, session)
    history = JobHistoryRepository(settings.history_path)
    resumes: dict[str, dict] = {}  # resume_id -> {"path", "name", "profile"}
    saved = session.resume()
    if saved:  # the resume from before a restart still works without uploading it again
        resumes[saved["resume_id"]] = {"path": saved["path"], "name": saved["file_name"], "profile": saved["profile"]}

    ctx = ApiContext(
        settings=settings,
        output_dir=output_dir,
        upload_dir=upload_dir,
        data_dir=data_dir,
        manager=manager,
        history=history,
        session=session,
        scheduler=scheduler,
        analyzer=analyzer,
        resumes=resumes,
    )

    app = FastAPI(title="Job Search")

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    for module in (resume_routes, session_routes, search_routes, history_routes, schedule_routes, config_routes, reports_routes):
        app.include_router(module.build_router(ctx))

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


# ---------- `serve`: the web UI ----------

def _serve(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="app.main serve", description="Open the job search web UI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Don't open the browser automatically")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "pdfminer", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        app = create_app()
    except ValueError as error:  # e.g. a bad value in .env
        logging.error("%s", error)
        return 1

    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Job search is running at {url}\n  Press Ctrl+C to stop.\n")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


# ---------- `search`: one-off command-line search ----------

def build_search_parser() -> argparse.ArgumentParser:
    # Search options default to None so we can tell "not passed" apart and fall back to .env.
    parser = argparse.ArgumentParser(
        prog="app.main search",
        description="Find recent jobs that match your resume. Defaults for every search option can be set in .env.",
    )
    parser.add_argument("--resume", type=Path, help="Path to your resume (.pdf, .docx, .txt) [.env: RESUME]")
    parser.add_argument("--sites", nargs="+", choices=list(SCRAPERS), help="Job boards to search (default: all) [.env: SITES]")
    parser.add_argument(
        "--location", dest="locations", nargs="+",
        help="Where to search, e.g. --location Pune Bengaluru Remote (default: from your resume) [.env: LOCATION]",
    )
    parser.add_argument("--experience", help="Years of experience wanted, e.g. '1-2' or '3' [.env: EXPERIENCE]")
    parser.add_argument(
        "--only-new", action=argparse.BooleanOptionalAction,
        help="Only show jobs not shown in earlier searches [.env: ONLY_NEW]",
    )
    parser.add_argument(
        "--keywords", nargs="+", help="Override search phrases, e.g. --keywords 'Python Developer' 'SDET' [.env: KEYWORDS]"
    )
    parser.add_argument("--hours", type=int, help="Only jobs posted within this many hours (default 24) [.env: HOURS]")
    parser.add_argument("--max-pages", type=int, help="Result pages per search (default 3) [.env: MAX_PAGES]")
    parser.add_argument(
        "--results", nargs="+", metavar="N|SITE=N",
        help="Jobs to show per site, best first: '--results 15' for every site, or "
        "'--results linkedin=20 naukri=12' (default 10) [.env: RESULTS, RESULTS_LINKEDIN, ...]",
    )
    parser.add_argument("--top", type=int, help="Jobs per site to open and score (default 40) [.env: TOP]")
    parser.add_argument("--no-llm", action="store_true", help="Skip OpenAI; rank by keyword overlap instead")
    parser.add_argument("--show-browser", action="store_true", help="Watch the browser while it works")
    parser.add_argument("--no-open", action="store_true", help="Don't open the HTML report when finished")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Where to save reports")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")
    return parser


def build_search_options(args: argparse.Namespace, settings: Settings) -> SearchOptions:
    """Merge command-line options over .env defaults. Raises ValueError for unusable combinations."""

    def pick(cli_value, env_value):
        return env_value if cli_value is None else cli_value

    resume = pick(args.resume, settings.resume)
    if resume is None:
        raise ValueError("no resume given: pass --resume my_resume.pdf, or set RESUME=... in .env")

    sites = pick(args.sites, settings.sites) or list(SCRAPERS)
    unknown = [site for site in sites if site not in SCRAPERS]
    if unknown:
        raise ValueError(f"unknown site(s) {', '.join(unknown)} in SITES; available: {', '.join(SCRAPERS)}")

    keywords = pick(args.keywords, settings.keywords)
    if args.no_llm and not keywords:
        raise ValueError("--no-llm needs keywords (--keywords or KEYWORDS in .env), since there is no AI to read your resume")

    experience = parse_experience_option(args.experience) if args.experience else settings.experience

    results, results_per_site = settings.results, dict(settings.results_per_site)
    if args.results:
        cli_results, cli_per_site = parse_results(args.results)
        if cli_results is not None:
            # A plain number on the command line applies to every site for this run.
            results, results_per_site = cli_results, {}
        results_per_site.update(cli_per_site)
    unknown = [site for site in results_per_site if site not in SCRAPERS]
    if unknown:
        raise ValueError(f"unknown site(s) {', '.join(unknown)} in results; available: {', '.join(SCRAPERS)}")

    top = pick(args.top, settings.top)
    hours = pick(args.hours, settings.hours)
    max_pages = pick(args.max_pages, settings.max_pages)
    numbers = [("--results", results), ("--top", top), ("--hours", hours), ("--max-pages", max_pages)]
    numbers += [(f"results for {site}", count) for site, count in results_per_site.items()]
    for name, value in numbers:
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    return SearchOptions(
        resume_path=resume,
        sites=sites,
        locations=pick(args.locations, settings.locations),
        keywords=keywords,
        experience=experience,
        only_new=pick(args.only_new, settings.only_new),
        history_path=settings.history_path,
        cache_path=settings.cache_path,
        parallel_sites=settings.parallel_sites,
        origin="cli",
        max_age_hours=hours,
        max_pages=max_pages,
        top_n=top,
        results=results,
        results_per_site=results_per_site,
        use_llm=not args.no_llm,
        headless=not args.show_browser,
        output_dir=args.output_dir,
    )


def parse_results(values: list[str]) -> tuple[int | None, dict[str, int]]:
    """Split ['15'] / ['linkedin=20', 'naukri=12'] into (all-sites count, per-site counts)."""
    everywhere, per_site = None, {}
    for value in values:
        site, _, count = value.rpartition("=")
        try:
            number = int(count)
        except ValueError:
            raise ValueError(f"--results expects a number or SITE=NUMBER, got {value!r}") from None
        if site:
            per_site[site.strip().lower()] = number
        else:
            everywhere = number
    return everywhere, per_site


def print_search_summary(result) -> None:
    profile = result.profile
    if profile.summary:
        print(f"\nProfile: {profile.summary}")
    if profile.skills:
        print(f"Skills:  {', '.join(profile.skills[:10])}")

    for site, jobs in result.top_by_site.items():
        print(f"\n{site.title()} - top {len(jobs)}:")
        for rank, job in enumerate(jobs, start=1):
            score = "--" if job.score is None else f"{job.score:>3}"
            print(f"  {rank:>2}. [{score}] {job.title} - {job.company} ({job.location}, {job.posted_text or 'recent'})")
            if job.matched_skills:
                print(f"            Skills: {', '.join(job.matched_skills)}")
            print(f"            {job.url}")
    print(f"\nReport: {result.html_path}")
    print(f"All {len(result.jobs)} scored jobs: {result.xlsx_path}")


def _search(argv: list[str] | None) -> int:
    parser = build_search_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        options = build_search_options(args, settings)
    except ValueError as error:
        parser.error(str(error))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party HTTP chatter out of normal output.
    for noisy in ("httpx", "openai", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        result = run(options, settings)
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        return 1
    except OpenAIError as error:
        logging.error("OpenAI request failed: %s", error)
        return 1

    print_search_summary(result)
    if not args.no_open:
        webbrowser.open(result.html_path.resolve().as_uri())
    return 0


# ---------- `daily`: run the saved daily search ----------

def _daily(_argv: list[str] | None) -> int:
    settings = load_settings()
    data_dir = settings.history_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(data_dir / LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
    )
    for noisy in ("httpx", "openai", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    schedule = load_schedule(data_dir)
    if schedule is None:
        log.error("No daily search is saved. Set one up in the web UI first.")
        return 1

    started = datetime.now().isoformat(timespec="seconds")
    log.info("Daily search started")
    options = options_from_dict(schedule.search)
    options.origin = "daily"
    options.parallel_sites = settings.parallel_sites
    try:
        result = run(options, settings)
    except Exception as error:  # record any failure so the UI can show it
        log.exception("Daily search failed")
        save_state(data_dir, last_run=started, status="error", error=str(error) or error.__class__.__name__)
        return 1

    shown = sum(len(jobs) for jobs in result.top_by_site.values())
    new = sum(1 for jobs in result.top_by_site.values() for job in jobs if job.is_new)
    save_state(
        data_dir, last_run=started, status="done", jobs=shown, new=new, report=str(Path(result.html_path).resolve()),
        duration_s=result.stats.duration_s, cost_usd=result.stats.cost_usd,
    )
    log.info("Daily search finished: %d jobs (%d new). Report: %s", shown, new, result.html_path)
    if schedule.open_report:
        webbrowser.open(Path(result.html_path).resolve().as_uri())
    try:
        cleanup(settings.keep_days, data_dir, Path(options.resume_path).parent, options.output_dir, settings.cache_path)
    except Exception:  # housekeeping must never fail the daily run
        log.exception("Cleanup failed")
    return 0


# ---------- dispatch ----------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("serve", "search", "daily"):
        print("usage: python -m app.main {serve,search,daily} ...", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "serve":
        return _serve(rest)
    if command == "search":
        return _search(rest)
    return _daily(rest)


if __name__ == "__main__":
    raise SystemExit(main())
