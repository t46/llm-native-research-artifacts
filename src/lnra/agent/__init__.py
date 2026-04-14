"""Agent interface for operating on LLM-native research artifacts.

The core innovation: query(), compose(), diff() operations that let
AI agents programmatically reason over structured scientific knowledge.
"""

from .interface import ArtifactAgent

__all__ = ["ArtifactAgent"]
