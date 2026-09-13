"""
Central configuration constants for Buy or Wait? AI financial decision agent.
"""

from __future__ import annotations
from pathlib import Path

import os

# Repository Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# Automatically load .env environment file
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_IMAGES_DIR = DATASET_DIR / "media" / "images"
OUTPUT_CSV_PATH = REPO_ROOT / "output.csv"
USAGE_REPORT_PATH = REPO_ROOT / "evaluation" / "usage_report.md"
CODE_USAGE_REPORT_PATH = REPO_ROOT / "code" / "evaluation" / "usage_report.md"
LOG_FILE_PATH = REPO_ROOT / "log.txt"

# Dataset Files
REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"

# Model & Execution Parameters
FORECAST_DAYS = 90
OCR_CONFIDENCE_THRESHOLD = 0.80
MAX_CONCURRENCY = 5

SUPPORTED_CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}

DEFAULT_VLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
