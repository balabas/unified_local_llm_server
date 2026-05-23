"""Inference pipelines sub-package.

Contains stateless pipeline classes instantiated once by ``LLMProviderPool``
and invoked during each call in ``pool._call()``:

- ``json_fix``   — structured JSON output with automatic repair and retry
- ``loop_guard`` — repetition-loop detection and corrective retry
- ``tool``       — tool-call normalization, execution, and message construction
"""
from .json_fix import JsonFixPipeline
from .loop_guard import LoopGuardPipeline, check_loop
from .tool import ToolPipeline

__all__ = ["JsonFixPipeline", "LoopGuardPipeline", "ToolPipeline", "check_loop"]
