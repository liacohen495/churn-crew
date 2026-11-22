"""Shared LLM settings. temperature=0 + fixed seed keep agent text as repeatable as possible."""
import os

from crewai import LLM

from .config import SEED


def get_llm() -> LLM:
    return LLM(
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        temperature=0,
        seed=SEED,
    )
