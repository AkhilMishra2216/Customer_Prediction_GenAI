"""
Streamlit Cloud entrypoint.

This file allows deployment platforms configured to run `streamlit.py`
while reusing the main application defined in `app.py`.
"""

from app import *  # noqa: F401,F403
