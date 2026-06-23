"""
services/__init__.py
────────────────────
Package marker. Imports nothing at module level — services are lazy-loaded
by the Flask app factory so the model isn't loaded during test imports.
"""
