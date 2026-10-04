"""What the page needs to render its form: sites, defaults, location choices."""

from fastapi import APIRouter

from app.api.dependencies import ApiContext
from app.config import LOCATION_CHOICES
from app.services.reporting import SITE_NAMES
from app.services.resume_service import SUPPORTED_SUFFIXES
from app.services.scraping import SCRAPERS


def build_router(ctx: ApiContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def config():
        settings = ctx.settings
        return {
            "sites": [
                {
                    "id": site,
                    "name": SITE_NAMES.get(site, site.title()),
                    "results": settings.results_per_site.get(site, settings.results),
                }
                for site in SCRAPERS
            ],
            "enabled_sites": settings.sites or list(SCRAPERS),
            "location_choices": LOCATION_CHOICES,
            "defaults": {
                "locations": settings.locations,
                "keywords": settings.keywords,
                "hours": settings.hours,
                "experience": list(settings.experience) if settings.experience else None,
                "only_new": settings.only_new,
                "max_pages": settings.max_pages,
                "top": settings.top,
                "parallel_sites": settings.parallel_sites,
            },
            "has_api_key": bool(settings.openai_api_key),
            "running_job": ctx.manager.running.id if ctx.manager.running else None,
            "resume_types": list(SUPPORTED_SUFFIXES),
            "history": ctx.history.counts(),
        }

    return router
