"""Paths and environment for Day 7."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
# Fall back to Day 6 .env for DATABASE_URL / GEMINI_API_KEY when Day 7 .env is incomplete.
_day6_starter = Path(os.environ.get(
    'DAY6_STARTER',
    '/Users/user/ai20k/vinuni-codelab/course/week01/day06/lab/work/'
    'K5-L3B-DAY06-HuynhVanCuong-2A202603866-DataIngestion/starter',
))
if (_day6_starter / '.env').exists():
    load_dotenv(_day6_starter / '.env', override=False)

DAY6_STARTER = Path(os.environ.get('DAY6_STARTER', str(_day6_starter)))
DAY6_STRATEGY = os.environ.get('DAY6_STRATEGY', 'B_struct')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash-lite').strip()
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
EMBEDDING_DIM = 384

DATA_DIR = ROOT / 'data'
CHUNKS_PATH = DATA_DIR / 'chunks.json'
QUESTIONS_ANSWERABLE = DATA_DIR / 'questions_answerable.json'
QUESTIONS_PATH = DATA_DIR / 'questions.json'
AGENT_CASES_PATH = DATA_DIR / 'agent_cases.json'
RESULTS_DIR = ROOT / 'starter' / 'results'


def ensure_day6_on_path() -> Path:
    starter = DAY6_STARTER.resolve()
    if not starter.exists():
        raise FileNotFoundError(f'DAY6_STARTER không tồn tại: {starter}')
    path = str(starter)
    if path not in sys.path:
        sys.path.insert(0, path)
    return starter


def require_database_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError('Thiếu DATABASE_URL (điền .env Day 7 hoặc Day 6).')
    return DATABASE_URL
