"""The Anomalica assimilator.

Integrates per-record digest files into the unified knowledge graph: entity
resolution, claim accumulation, corroboration, and evidence scoring. Maintains
the graph incrementally; the SQLite database is the public, rebuildable dataset.
"""
