"""Root Flask entrypoint for Render and other deployments.

This file allows `gunicorn app:app` to work even when the real app lives in backend/app.py.
"""

import sys
from pathlib import Path

# Ensure backend directory is on the import path.
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
