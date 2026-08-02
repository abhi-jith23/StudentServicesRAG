from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
CLEAN_HTML_DIR = DATA_DIR / "cleaned" / "html"
CLEAN_PDF_DIR = DATA_DIR / "cleaned" / "pdf"
REVIEW_DIR = DATA_DIR / "review"

SOURCES_CSV = CATALOG_DIR / "sources.csv"
DISCOVERED_LINKS_CSV = CATALOG_DIR / "discovered_links.csv"
FETCH_MANIFEST_CSV = CATALOG_DIR / "fetch_manifest.csv"
COLLECTION_ISSUES_CSV = REVIEW_DIR / "collection_issues.csv"

ALLOWED_DOMAIN_SUFFIXES = (
    "uni.lu",
    "guichet.public.lu",
    "cns.public.lu",
    "mengstudien.public.lu",
)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 15
DEFAULT_READ_TIMEOUT_SECONDS = 60
DEFAULT_PLAYWRIGHT_TIMEOUT_MS = 45_000
DEFAULT_MAX_DOWNLOAD_BYTES = 75 * 1024 * 1024
DEFAULT_DELAY_SECONDS = 1.0

HTML_MIN_TEXT_LENGTH = 300
PDF_MIN_TEXT_LENGTH = 500
PDF_SHORT_PAGE_THRESHOLD = 40