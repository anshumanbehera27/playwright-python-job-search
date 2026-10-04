"""Registry of available job-board scrapers. Add new boards here (order = report order)."""

from app.services.scraping.base import BaseScraper
from app.services.scraping.browser import SiteSession, block_assets
from app.services.scraping.indeed import IndeedScraper
from app.services.scraping.linkedin import LinkedInScraper
from app.services.scraping.naukri import NaukriScraper

SCRAPERS: dict[str, type[BaseScraper]] = {
    LinkedInScraper.name: LinkedInScraper,
    NaukriScraper.name: NaukriScraper,
    IndeedScraper.name: IndeedScraper,
}

__all__ = ["SCRAPERS", "BaseScraper", "SiteSession", "block_assets"]
