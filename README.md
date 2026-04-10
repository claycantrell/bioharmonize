# bioharmonize

QC, repair, and readiness checking for biological analysis objects.

bioharmonize normalizes messy metadata on DataFrames and AnnData objects, validates
it against configurable profiles, and tells you whether a dataset is ready for
downstream tasks like clustering, differential expression, or integration.

## Install

```bash
pip install bioharmonize
pip install bioharmonize[anndata]   # AnnData support
pip install bioharmonize[cli]       # CLI
```

Requires Python 3.10+.

## Quickstart

### Repair metadata

```python
import bioharmonize as bh

report = bh.repair(df, profile="single_cell_human")
print(report.summary())
```

`repair()` renames columns (`celltype` -> `cell_type`), normalizes values
(`m` -> `male`, `ctrl` -> `control`), coerces dtypes, and validates. Every
change is recorded in `report.changes`.

### Inspect without modifying

```python
report = bh.inspect(df)
# report.changes shows what *would* be renamed/normalized
# report.issues shows validation findings
```

### Validate only

```python
report = bh.validate(df, level="strict")
for issue in report.issues:
    print(f"{issue.severity}: {issue.message}")
```

Three validation levels: `minimal`, `standard` (default), `strict`.

### Check task readiness

```python
report = bh.preflight(adata, "differential_expression")
# Checks: required columns present, >=2 condition groups,
# replicates per condition, X matrix has raw counts, etc.
```

Built-in tasks: `clustering`, `differential_expression`, `integration`,
`cell_type_annotation`. Each checks task-specific metadata and (for AnnData)
matrix-level properties.

### AnnData workflows

```python
import scanpy as sc
import bioharmonize as bh

adata = sc.read_h5ad("study.h5ad")

# Option 1: repair obs directly
report = bh.repair(adata)
repaired_adata = report.adata

# Option 2: patch helper
new_adata, report = bh.patch_anndata(adata)
```

When passed an AnnData, `inspect()` and `validate()` also run structural sanity
checks (shape consistency, duplicate cell IDs, all-zero genes, layer presence).

## The Report object

Every API function returns a `Report`:

```python
report.cleaned       # repaired DataFrame
report.adata         # repaired AnnData (if input was AnnData)
report.changes       # list of Change objects (rename, normalize, coerce)
report.issues        # list of Issue objects (error, warning, info)
report.readiness     # dict: {task_name: "ready"|"warning"|"not_ready"}
report.summary()     # formatted text overview
report.save("out/")  # writes cleaned.csv, issues.csv, changes.csv, summary.txt
```

## Profiles

A profile defines canonical column names, aliases, value normalizers, and
required columns per validation level. The built-in profile is `single_cell_human`.

**What it normalizes:**

| Messy input | Canonical output |
|---|---|
| `celltype`, `cell_type_annotation` | `cell_type` |
| `patient`, `donor` | `donor_id` |
| `gender` | `sex` |
| `m`, `f` | `male`, `female` |
| `ctrl`, `healthy` | `control` |

Custom profiles:

```python
from bioharmonize import Profile, repair
from bioharmonize.specs import ColumnSpec

my_profile = Profile(
    name="my_study",
    canonical_columns={"gene_symbol": ColumnSpec(name="gene_symbol", dtype="string")},
    column_aliases={"gene": "gene_symbol"},
    value_normalizers={},
)
report = repair(df, profile=my_profile)
```

Override mappings at call time:

```python
report = bh.repair(df, column_map={"Patient_ID": "donor_id"},
                       value_maps={"sex": {"M": "male", "F": "female"}})
```

## CLI

```bash
bioharmonize inspect metadata.csv
bioharmonize repair metadata.csv --validation strict -o output/
bioharmonize validate metadata.csv --level strict
bioharmonize patch study.h5ad -o study_harmonized.h5ad
bioharmonize preflight metadata.csv differential_expression
```

## License

MIT
