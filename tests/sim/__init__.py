"""Heavyweight sim policy tests.

These are NOT part of the default ``unittest`` / ``pytest`` discovery — they
boot MuJoCo, load each registered policy's ONNX/JIT model, and drive the full
pipeline for several seconds. See ``AGENTS.md`` for the intended workflow.
"""
