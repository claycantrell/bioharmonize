# Real Single-Cell Metadata Test Fixtures

Sampled obs metadata from public GEO single-cell RNA-seq datasets, selected to
span a range of metadata quality levels for integration testing of bioharmonize.

Each CSV contains 200 rows randomly sampled (stratified by cell type where
available) from the original dataset.

## Datasets

### 1. `geo_lung_cancer_obs.csv` — Clean

**Source:** [GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907)
Kim et al., "Single-cell RNA sequencing demonstrates the molecular and cellular
reprogramming of metastatic lung adenocarcinoma." *Nature Communications* (2020).

**License:** Public domain (GEO/NCBI)

**Why clean:** Uses near-standard column names (`Cell_type`, `Cell_subtype`).
Values are human-readable full words. Minimal QC columns.

| Column | Maps to | Notes |
|--------|---------|-------|
| `Cell_type` | `cell_type` | Full names: "T lymphocytes", "Myeloid cells" |
| `Sample_Origin` | `tissue` | Abbreviated: "nLung", "tLung", "mBrain", "PE" |
| `Sample` | `sample_id` | e.g. "LUNG_N01" |
| `Cell_type.refined` | — | Finer annotation |
| `Cell_subtype` | — | Subtype labels |

**Missing columns:** sex, disease/condition, donor_id, assay

---

### 2. `geo_kidney_myeloid_obs.csv` — Moderately Messy

**Source:** [GSE154763](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154763)
Cheng et al., "A pan-cancer single-cell transcriptional atlas of tumor
infiltrating myeloid cells." *Cell* (2021).

**License:** Public domain (GEO/NCBI)

**Why moderately messy:** Non-standard column names throughout. Tissue values
are single-letter abbreviations. Cell type column embeds cluster IDs.

| Column | Maps to | Notes |
|--------|---------|-------|
| `MajorCluster` | `cell_type` | Encoded: "M07_Mono_CD16", "M04_cDC2_CD1C" |
| `patient` | `donor_id` | e.g. "P20181217" |
| `tissue` | `tissue` | Single letter: "N" (normal), "T" (tumor) |
| `cancer` | `disease` | All "KIDNEY" |
| `tech` | `assay` | All "10X5" |
| `source` | — | Lab source |
| `percent_mito`, `n_counts`, `n_genes` | — | QC metrics mixed in |

**Missing columns:** sex, condition, sample_id

---

### 3. `geo_crc_leukocyte_obs.csv` — Messy

**Source:** [GSE146771](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146771)
Zhang et al., "Single-cell analyses inform mechanisms of myeloid-targeted
therapies in colon cancer." *Cell* (2020).

**License:** Public domain (GEO/NCBI)

**Why messy:** 23 columns with most being QC metrics. Cell type buried in
`Global_Cluster`. Tissue is single-letter coded. Dot-separated column names.

| Column | Maps to | Notes |
|--------|---------|-------|
| `Global_Cluster` | `cell_type` | "CD4 T cell", "Myeloid cell", "B cell" |
| `Sub_Cluster` | — | "hT04_CD4-TCF7" etc. |
| `Tissue` | `tissue` | "N", "T", "P" (peripheral blood) |
| `Platform` | `assay` | All "10X" |
| `Sample` | `sample_id` | e.g. "P1025" |
| `raw.nUMI`, `raw.nGene`, `filter.nUMI`, ... | — | 8 QC metric columns |
| `Global_tSNE_1`, `Sub_tSNE_1`, ... | — | 6 embedding columns |

**Missing columns:** sex, disease/condition, donor_id

---

### 4. `geo_hsc_facs_obs.csv` — Very Messy

**Source:** [GSE148884](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148884)
Calvanese et al., "Mapping human haematopoietic stem cells from haemogenic
endothelium to birth." *Nature* (2022).

**License:** Public domain (GEO/NCBI)

**Why very messy:** No cell type column at all. FACS marker intensities serve as
the main annotation (CD34, CD38, CD45RA, CD49f, CD90). Column names use
dot-separated R conventions. Completely different structure from typical scRNA-seq
obs metadata.

| Column | Maps to | Notes |
|--------|---------|-------|
| `Donor` | `donor_id` | "mPB-3", "BM1" |
| `Tissue` | `tissue` | Full text: "Mobilized Peripheral Blood", "Bone Marrow" |
| `CD34`, `CD38`, `CD45RA`, `CD49f`, `CD90` | — | FACS marker intensities |
| `Cycling` | — | "not_cycling" / "cycling" |
| `pct_counts_MT` | — | Mitochondrial % |
| `S.Score`, `G2M.Score` | — | Cell cycle scores |

**Missing columns:** cell_type, sex, disease/condition, sample_id, assay
