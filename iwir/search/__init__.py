"""Pluggable search providers. Add a class and register it in ``build_providers``."""

from iwir.search.chain import build_services, search_chain

__all__ = ["build_services", "search_chain"]
