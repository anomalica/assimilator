"""Command-line interface for the assimilator.

The assimilator integrates per-record digest files (from the digester) into the
unified Anomalica knowledge graph. The import / matching / consolidate / scoring /
database logic is being migrated out of the digester; the commands below are the
planned interface and are wired up as the modules land here.
"""

import click


@click.group()
def cli() -> None:
    """Build and maintain the Anomalica knowledge graph from digest files."""


@cli.command()
@click.argument("digests_dir")
def assimilate(digests_dir: str) -> None:
    """Integrate digest files into the knowledge graph (incremental)."""
    raise SystemExit(
        "not yet implemented - migrating import/matching/consolidate/scoring from the digester"
    )


@cli.command()
def rebuild() -> None:
    """Rebuild the knowledge graph deterministically from all digests."""
    raise SystemExit("not yet implemented")


@cli.command()
def status() -> None:
    """Report graph state: node / claim / corroboration counts."""
    raise SystemExit("not yet implemented")


if __name__ == "__main__":
    cli()
