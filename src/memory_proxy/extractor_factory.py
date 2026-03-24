from __future__ import annotations

from .config import ProxySettings
from .extractor import RuleBasedExtractor


def build_memory_extractor(settings: ProxySettings) -> RuleBasedExtractor:
    del settings
    # Compression stays deterministic at runtime: no LLM-assisted extractor is
    # selected from environment flags.
    return RuleBasedExtractor()
