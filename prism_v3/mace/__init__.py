"""MACE-RCA package.

The object-graph utilities are used by non-LLM offline analysis modules.  Keep
the package importable even when the optional LLM client is not installed.
"""

try:
    from .pipeline import MACEConfig, MACEPipeline
except ModuleNotFoundError:
    MACEConfig = None
    MACEPipeline = None

__all__ = ["MACEConfig", "MACEPipeline"]
