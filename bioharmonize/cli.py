from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    try:
        import click
    except ImportError:
        print(
            "The bioharmonize CLI requires click. Install with: pip install bioharmonize[cli]",
            file=sys.stderr,
        )
        sys.exit(1)

    _build_cli()(argv)


def _build_cli():
    import click

    from .api import inspect as _inspect
    from .api import preflight as _preflight
    from .api import repair as _repair
    from .api import validate as _validate
    from .io import read_obs

    # keep backward-compat imports for old commands
    from .api import clean_obs, validate_obs

    @click.group()
    def cli():
        """bioharmonize - Normalize and validate biological study metadata."""
        pass

    # ── New top-level commands ───────────────────────────────────────────

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    def inspect(path: str, profile: str):
        """Analyse metadata and show diagnostics without modifying data."""
        df = read_obs(path)
        report = _inspect(df, profile=profile)
        click.echo(report.summary())

    @cli.command("repair")
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    @click.option(
        "--validation",
        default="standard",
        type=click.Choice(["minimal", "standard", "strict"]),
        help="Validation level",
    )
    @click.option("--output", "-o", default=".", help="Output directory")
    def repair_cmd(path: str, profile: str, validation: str, output: str):
        """Repair metadata: rename columns, normalise values, validate."""
        df = read_obs(path)
        report = _repair(df, profile=profile, validation=validation)
        out_dir = Path(output)
        report.save(out_dir)
        click.echo(report.summary())

    @cli.command("preflight")
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    @click.option(
        "--validation",
        default="standard",
        type=click.Choice(["minimal", "standard", "strict"]),
        help="Validation level",
    )
    def preflight_cmd(path: str, profile: str, validation: str):
        """Dry-run of repair: show what would change without saving."""
        df = read_obs(path)
        report = _preflight(df, profile=profile, validation=validation)
        click.echo(report.summary())

    # ── Backward-compatible commands ────────────────────────────────────

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    @click.option(
        "--validation",
        default="standard",
        type=click.Choice(["minimal", "standard", "strict"]),
        help="Validation level",
    )
    @click.option("--output", "-o", default=".", help="Output directory")
    def clean(path: str, profile: str, validation: str, output: str):
        """Clean and normalize a metadata table."""
        df = read_obs(path)
        report = clean_obs(df, profile=profile, validation=validation)
        out_dir = Path(output)
        report.save(out_dir)
        click.echo(report.summary())

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    @click.option(
        "--level",
        default="standard",
        type=click.Choice(["minimal", "standard", "strict"]),
        help="Validation level",
    )
    def validate(path: str, profile: str, level: str):
        """Validate a metadata table without modifying it."""
        df = read_obs(path)
        report = validate_obs(df, profile=profile, level=level)
        click.echo(report.summary())
        errors = [i for i in report.issues if i.severity == "error"]
        if errors and level == "strict":
            sys.exit(1)

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--profile", default="single_cell_human", help="Validation profile name")
    @click.option("--output", "-o", default=None, help="Output .h5ad path")
    @click.option(
        "--validation",
        default="standard",
        type=click.Choice(["minimal", "standard", "strict"]),
        help="Validation level",
    )
    def patch(path: str, profile: str, output: str | None, validation: str):
        """Patch an AnnData .h5ad file's obs metadata."""
        try:
            import anndata
        except ImportError:
            click.echo(
                "anndata is required for the patch command. "
                "Install with: pip install bioharmonize[anndata]",
                err=True,
            )
            sys.exit(1)

        from .anndata import patch_anndata

        ad = anndata.read_h5ad(path)
        new_ad, report = patch_anndata(ad, profile=profile, validation=validation)
        click.echo(report.summary())

        out_path = output or path.replace(".h5ad", "_harmonized.h5ad")
        new_ad.write_h5ad(out_path)
        click.echo(f"\nPatched file written to: {out_path}")

    return cli
