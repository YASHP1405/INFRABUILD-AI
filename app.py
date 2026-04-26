"""Root Flask entrypoint for Render and other deployments.

This file allows `gunicorn app:app` to work by importing the Flask app from backend/app.py.
"""

from backend.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
