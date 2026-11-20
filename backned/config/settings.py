"""
Centralised configuration — loads .env once and exposes all
environment variables as module-level constants.

Usage
-----
    from backned.config.settings import AZURE_AI_ENDPOINT, API_KEY, ...
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

#  Locate and load the .env at the project root 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → Invoice_Auditor/
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

#  Azure OpenAI 
AZURE_AI_ENDPOINT: str = os.getenv("AZURE_AI_ENDPOINT", "")
API_VERSION: str = os.getenv("API_VERSION", "2024-05-01-preview")
API_KEY: str = os.getenv("API_KEY", "")
DEPLOYMENT_NAME: str = os.getenv("DEPLOYMENT_NAME", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")

#  LangFuse Observability 
LANGFUSE_PUBLIC_KEY: str  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str  = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED: bool    = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
