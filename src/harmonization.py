#!/usr/bin/env python3
"""Multi-omics harmonization pipeline (autism vs. matched controls).

Builds three sample-aligned matrices from heterogeneous inputs in ``data/``:

    * data/metagenomics.tsv   samples x taxa        (Bracken relative abundance)
    * data/metabolomics.tsv   samples x pathways    (HUMAnN pathabundance)
    * data/clinical.tsv       samples x variables   (SPSS clinical records)

The canonical sample identifier is the *Novogen* ID (e.g. ``A1`` autism,
``K1`` control) used in the metagenomic and metabolomic files. Clinical records
are keyed by a separate "Šifra" code and are linked to Novogen IDs through the
mapping workbook ``Sifrarnik_kakice.xlsx``.

All three deliverables share the *same* set of sample IDs in the *same* order
(the order in which samples appear in ``merged_bracken.tsv``), as required for
downstream multi-omics analysis.

Design decisions (see CLAUDE.md / PLAN.md and the interactive choices made
during development):

  * Metabolomics matrix  -> HUMAnN ``pathabundance`` files, community-level
    (unstratified) features only. The folder is named "metabolomics_raw_data"
    but in fact contains HUMAnN functional profiles; pathabundance is the most
    metabolism-relevant, manageably sized representation.
  * Clinical duplicates  -> 6 clinical codes occur twice with conflicting
    values. We KEEP THE FIRST occurrence of each and report every dropped row
    in detail (logs + PDF report).

A full PDF report with visualizations is written to ``data/harmonization_report.pdf``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

BRACKEN_TSV = os.path.join(DATA, "merged_bracken.tsv")
CLINICAL_SAV = os.path.join(DATA, "clinical.sav")
MAPPING_XLSX = os.path.join(DATA, "Sifrarnik_kakice.xlsx")
METAB_DIR = os.path.join(DATA, "metabolomics_raw_data")

OUT_METAGENOMICS = os.path.join(DATA, "metagenomics.tsv")
OUT_METABOLOMICS = os.path.join(DATA, "metabolomics.tsv")
OUT_CLINICAL = os.path.join(DATA, "clinical.tsv")
OUT_CLINICAL_FULL = os.path.join(DATA, "clinical_full.tsv")  # raw SPSS -> TSV
OUT_REPORT_PDF = os.path.join(DATA, "harmonization_report.pdf")

BRACKEN_FRAC_SUFFIX = ".bracken_frac"
CLINICAL_ID_COL = "Šifra"
METAB_FILE_KIND = "pathabundance"  # which HUMAnN file type feeds metabolomics

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("harmonization")


# --------------------------------------------------------------------------- #
# Provenance: a record of every harmonization decision, for logs + PDF report
# --------------------------------------------------------------------------- #
@dataclass
class Provenance:
    """Accumulates counts and notes so the PDF report can be fully data-driven."""

    n_bracken: int = 0
    n_metabolomics: int = 0
    n_clinical_rows: int = 0
    n_clinical_unique: int = 0
    n_mapping_rows: int = 0
    n_novogen_with_clinical: int = 0
    n_final: int = 0

    n_taxa: int = 0
    n_pathways: int = 0
    n_clinical_vars: int = 0

    n_autism: int = 0
    n_control: int = 0

    # samples present in omics but lacking usable clinical data
    dropped_no_clinical: list = field(default_factory=list)
    # conflicting duplicate clinical codes (kept first, dropped the rest)
    clinical_dup_notes: list = field(default_factory=list)
    # clinical codes that could not be linked to any Novogen ID
    unmapped_clinical_codes: list = field(default_factory=list)


PROV = Provenance()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_code(value) -> str | None:
    """Normalize a clinical sample code to a canonical ``ka<n>`` / ``kk<n>`` form.

    Handles the inconsistencies observed between ``clinical.sav`` and the mapping
    workbook: a ``2024-`` year prefix, inconsistent zero-padding (``ka2`` vs
    ``ka02``), stray whitespace/case, and trailing free-text notes such as
    ``"2024-kk203 - zamena za kk93"`` (only the leading code is used).

    Returns ``None`` when no ``ka``/``kk`` code can be extracted.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    m = re.search(r"(ka|kk)\s*0*(\d+)", text)
    if not m:
        return None
    return f"{m.group(1)}{int(m.group(2))}"


def novogen_group(novogen_id: str) -> str:
    """Autism ('A...') vs control ('K...') from the Novogen ID prefix."""
    return "autism" if novogen_id.upper().startswith("A") else "control"


# --------------------------------------------------------------------------- #
# Step 1 — Metagenomics (Bracken)
# --------------------------------------------------------------------------- #
def load_metagenomics() -> pd.DataFrame:
    """Parse ``merged_bracken.tsv`` into a samples x taxa relative-abundance matrix.

    Only the ``*.bracken_frac`` columns are used (relative abundance); the
    ``.bracken_frac`` suffix is stripped to recover the Novogen sample ID. The
    original column order is preserved and later used as the canonical sample
    order for all deliverables.
    """
    log.info("[metagenomics] reading %s", os.path.relpath(BRACKEN_TSV, ROOT))
    df = pd.read_csv(BRACKEN_TSV, sep="\t", low_memory=False)

    frac_cols = [c for c in df.columns if c.endswith(BRACKEN_FRAC_SUFFIX)]
    sample_ids = [c[: -len(BRACKEN_FRAC_SUFFIX)] for c in frac_cols]
    log.info("[metagenomics] %d taxa x %d samples", len(df), len(frac_cols))

    # taxa names index -> select frac columns -> transpose to samples x taxa
    mat = df.set_index("name")[frac_cols].T
    mat.index = sample_ids
    mat.index.name = "sample_id"
    mat = mat.astype(float)

    PROV.n_bracken = mat.shape[0]
    PROV.n_taxa = mat.shape[1]
    return mat


# --------------------------------------------------------------------------- #
# Step 2 — Clinical (SPSS .sav -> TSV) + mapping to Novogen IDs
# --------------------------------------------------------------------------- #
def convert_clinical() -> pd.DataFrame:
    """Read ``clinical.sav`` (SPSS), write the full conversion to TSV, and attach
    a normalized join key.

    Returns the full clinical table with an extra ``_code_norm`` column.
    """
    log.info("[clinical] reading SPSS file %s", os.path.relpath(CLINICAL_SAV, ROOT))
    df, _meta = pyreadstat.read_sav(CLINICAL_SAV)
    df["_code_norm"] = df[CLINICAL_ID_COL].map(normalize_code)

    df.to_csv(OUT_CLINICAL_FULL, sep="\t", index=False)
    log.info(
        "[clinical] %d rows x %d variables -> %s",
        df.shape[0],
        df.shape[1] - 1,  # exclude the helper column from the variable count
        os.path.relpath(OUT_CLINICAL_FULL, ROOT),
    )

    PROV.n_clinical_rows = df.shape[0]
    PROV.n_clinical_unique = df["_code_norm"].nunique(dropna=True)
    return df


def dedup_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve duplicate clinical codes by keeping the FIRST occurrence.

    The duplicates conflict substantially (some differ in hundreds of columns),
    so they cannot be safely merged. Every dropped row is reported in detail.
    """
    dup_mask = df["_code_norm"].duplicated(keep=False) & df["_code_norm"].notna()
    if dup_mask.any():
        log.warning(
            "[clinical] %d rows share a duplicated code; keeping first of each:",
            int(dup_mask.sum()),
        )
        for code, grp in df[dup_mask].groupby("_code_norm"):
            # number of columns that actually differ across the duplicate rows
            differing = int((grp.drop(columns=["_code_norm"]).nunique(dropna=False) > 1).sum())
            note = (
                f"{code}: {len(grp)} rows (orig Šifra={list(grp[CLINICAL_ID_COL])}), "
                f"{differing} columns differ -> kept first, dropped {len(grp) - 1}"
            )
            log.warning("[clinical]   %s", note)
            PROV.clinical_dup_notes.append(note)

    deduped = df.drop_duplicates(subset="_code_norm", keep="first")
    return deduped


def load_mapping() -> dict[str, str]:
    """Build a Novogen-ID -> normalized-clinical-code dictionary from the workbook.

    The workbook has two sheets (``ASD`` and ``Kontrole``). We keep only rows with
    a valid Novogen ID (``A``/``K`` followed by digits) and a parseable clinical
    code.
    """
    log.info("[mapping] reading %s", os.path.relpath(MAPPING_XLSX, ROOT))
    sheets = pd.read_excel(MAPPING_XLSX, sheet_name=None)

    mapping: dict[str, str] = {}
    n_rows = 0
    for sheet_name, sheet in sheets.items():
        for _, row in sheet.iterrows():
            n_rows += 1
            novogen = str(row["Novogen"]).strip()
            if not re.fullmatch(r"[AK]\d+", novogen, flags=re.IGNORECASE):
                continue  # skip blank / malformed Novogen IDs
            code = normalize_code(row["Sifra uzorka (klinicki podaci)"])
            if code is None:
                continue
            mapping[novogen.upper()] = code

    PROV.n_mapping_rows = n_rows
    log.info("[mapping] %d usable Novogen->clinical links", len(mapping))
    return mapping


# --------------------------------------------------------------------------- #
# Step 3 — Metabolomics (HUMAnN pathabundance)
# --------------------------------------------------------------------------- #
def _read_pathabundance(path: str) -> pd.Series:
    """Read one HUMAnN pathabundance file -> community-level pathway Series.

    Keeps only *unstratified* features (those without a ``|g__...`` species
    suffix), i.e. the per-pathway community totals.
    """
    s = pd.read_csv(path, sep="\t", header=0, names=["feature", "abundance"])
    s = s[~s["feature"].str.contains(r"\|", regex=True)]  # drop stratified rows
    return pd.Series(s["abundance"].values, index=s["feature"].values, dtype=float)


def load_metabolomics() -> pd.DataFrame:
    """Assemble a samples x pathways matrix from all pathabundance files.

    Features absent in a given sample are filled with 0 (HUMAnN omits zero rows).
    """
    files = sorted(
        f for f in os.listdir(METAB_DIR) if f.endswith(f"_{METAB_FILE_KIND}.tsv")
    )
    log.info("[metabolomics] reading %d %s files", len(files), METAB_FILE_KIND)

    per_sample: dict[str, pd.Series] = {}
    for fname in files:
        sample_id = fname.split("_kneaddata")[0]
        per_sample[sample_id] = _read_pathabundance(os.path.join(METAB_DIR, fname))

    # outer-join all samples on the feature axis, then orient samples x pathways
    mat = pd.DataFrame(per_sample).T
    mat = mat.fillna(0.0)
    mat.index.name = "sample_id"
    mat = mat.sort_index(axis=1)

    PROV.n_metabolomics = mat.shape[0]
    PROV.n_pathways = mat.shape[1]
    log.info("[metabolomics] %d samples x %d pathways", *mat.shape)
    return mat


# --------------------------------------------------------------------------- #
# Step 4 — Harmonization
# --------------------------------------------------------------------------- #
def harmonize(
    meta_g: pd.DataFrame,
    meta_b: pd.DataFrame,
    clinical: pd.DataFrame,
    mapping: dict[str, str],
):
    """Reduce all three datasets to the common sample set, identically ordered.

    The canonical order is the Bracken column order (``meta_g.index``), filtered
    to samples that are present in *all* of: metagenomics, metabolomics, and
    (via the mapping) clinical.
    """
    clinical_by_code = clinical.set_index("_code_norm")
    available_codes = set(clinical_by_code.index)

    # Novogen IDs that can be linked to an actual clinical record
    novogen_with_clinical = {
        nov for nov, code in mapping.items() if code in available_codes
    }
    PROV.n_novogen_with_clinical = len(novogen_with_clinical)

    bracken_order = list(meta_g.index)  # canonical order
    metab_set = set(meta_b.index)

    common, dropped = [], []
    for nov in bracken_order:
        if nov in metab_set and nov in novogen_with_clinical:
            common.append(nov)
        else:
            dropped.append(nov)

    PROV.dropped_no_clinical = [
        nov for nov in dropped if nov not in novogen_with_clinical
    ]
    PROV.n_final = len(common)
    log.info(
        "[harmonize] common samples: %d (dropped %d lacking clinical/metabolomics)",
        len(common),
        len(dropped),
    )

    # ----- align the three matrices to `common`, in identical order -----
    mg = meta_g.loc[common]

    mb = meta_b.loc[common]

    # clinical: map each Novogen ID -> its clinical code -> clinical row
    clin_rows = clinical_by_code.loc[[mapping[nov] for nov in common]].copy()
    clin_rows.insert(0, "sample_id", common)
    clin_rows.insert(1, "group", [novogen_group(n) for n in common])
    clin_rows = clin_rows.set_index("sample_id")
    clin_rows = clin_rows.drop(columns=["_code_norm"], errors="ignore")

    PROV.n_clinical_vars = clin_rows.shape[1]
    PROV.n_autism = sum(1 for n in common if novogen_group(n) == "autism")
    PROV.n_control = len(common) - PROV.n_autism

    return common, mg, mb, clin_rows


def validate(common, mg, mb, clin):
    """Hard guarantees: identical sample sets AND identical ordering everywhere."""
    log.info("[validate] checking sample alignment across all three matrices")
    assert list(mg.index) == common, "metagenomics order mismatch"
    assert list(mb.index) == common, "metabolomics order mismatch"
    assert list(clin.index) == common, "clinical order mismatch"
    assert len(set(common)) == len(common), "duplicate sample IDs in final set"
    log.info("[validate] OK — %d samples, identically ordered", len(common))


# --------------------------------------------------------------------------- #
# Step 5 — PDF report
# --------------------------------------------------------------------------- #
def _text_page(pdf: PdfPages, title: str, lines: list[str]):
    """Render a plain text page into the PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.07, 0.95, title, fontsize=16, fontweight="bold", va="top")
    body = "\n".join(lines)
    fig.text(0.07, 0.89, body, fontsize=9, va="top", family="monospace", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def build_report(common, mg, mb, clin):
    """Write a multi-page PDF documenting every harmonization step + visuals."""
    log.info("[report] writing %s", os.path.relpath(OUT_REPORT_PDF, ROOT))
    with PdfPages(OUT_REPORT_PDF) as pdf:
        # ---- Page 1: title + provenance summary --------------------------- #
        _text_page(
            pdf,
            "Multi-omics Harmonization Report",
            [
                "Autism vs. matched controls — sample-aligned multi-omics datasets",
                "=" * 64,
                "",
                "INPUT INVENTORY",
                f"  Metagenomic samples (Bracken)      : {PROV.n_bracken}",
                f"  Metabolomic samples (HUMAnN path.) : {PROV.n_metabolomics}",
                f"  Clinical rows (SPSS .sav)          : {PROV.n_clinical_rows}",
                f"    unique clinical codes            : {PROV.n_clinical_unique}",
                f"  Mapping workbook rows              : {PROV.n_mapping_rows}",
                "",
                "HARMONIZATION FUNNEL",
                f"  Novogen IDs linkable to clinical   : {PROV.n_novogen_with_clinical}",
                f"  FINAL common, aligned samples      : {PROV.n_final}",
                f"    autism                           : {PROV.n_autism}",
                f"    control                          : {PROV.n_control}",
                "",
                "DELIVERABLE DIMENSIONS (samples x features)",
                f"  metagenomics.tsv : {mg.shape[0]} x {mg.shape[1]} taxa (rel. abundance)",
                f"  metabolomics.tsv : {mb.shape[0]} x {mb.shape[1]} pathways (HUMAnN pathabundance, unstratified)",
                f"  clinical.tsv     : {clin.shape[0]} x {clin.shape[1]} variables",
                "",
                "KEY DECISIONS",
                "  * Metabolomics = HUMAnN pathabundance, community-level features.",
                "  * Clinical duplicates: kept FIRST occurrence (see dedicated page).",
                "  * Sample order = order of appearance in merged_bracken.tsv.",
                "  * Join key: clinical Šifra -> normalized ka/kk code -> Novogen ID.",
            ],
        )

        # ---- Page 2: harmonization funnel + group composition ------------- #
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 6))
        stages = ["Bracken", "Metabolomics", "Linkable\nclinical", "Final\naligned"]
        vals = [
            PROV.n_bracken,
            PROV.n_metabolomics,
            PROV.n_novogen_with_clinical,
            PROV.n_final,
        ]
        ax1.bar(stages, vals, color="#4C72B0")
        for i, v in enumerate(vals):
            ax1.text(i, v + 1, str(v), ha="center", fontsize=10)
        ax1.set_title("Sample counts through harmonization")
        ax1.set_ylabel("samples")

        ax2.bar(
            ["autism", "control"],
            [PROV.n_autism, PROV.n_control],
            color=["#C44E52", "#55A868"],
        )
        for i, v in enumerate([PROV.n_autism, PROV.n_control]):
            ax2.text(i, v + 0.5, str(v), ha="center", fontsize=10)
        ax2.set_title("Final cohort composition")
        ax2.set_ylabel("samples")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: metagenomics overview -------------------------------- #
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 6))
        # mean relative abundance of top 15 taxa
        top = mg.mean(axis=0).sort_values(ascending=False).head(15)[::-1]
        ax1.barh(range(len(top)), top.values, color="#4C72B0")
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels([t[:40] for t in top.index], fontsize=7)
        ax1.set_title("Top 15 taxa by mean relative abundance")
        ax1.set_xlabel("mean bracken_frac")

        # richness: number of detected taxa per sample
        richness = (mg > 0).sum(axis=1)
        ax2.hist(richness, bins=25, color="#8172B3")
        ax2.set_title("Taxa richness per sample")
        ax2.set_xlabel("number of detected taxa")
        ax2.set_ylabel("samples")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 4: metabolomics overview -------------------------------- #
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 6))
        pw_present = (mb > 0).sum(axis=1)
        ax1.hist(pw_present, bins=25, color="#CCB974")
        ax1.set_title("Pathways detected per sample")
        ax1.set_xlabel("number of non-zero pathways")
        ax1.set_ylabel("samples")

        # overall sparsity
        sparsity = float((mb == 0).mean().mean())
        prevalence = (mb > 0).mean(axis=0).sort_values(ascending=False)
        ax2.plot(range(len(prevalence)), prevalence.values, color="#CCB974")
        ax2.set_title(f"Pathway prevalence (matrix sparsity={sparsity:.1%})")
        ax2.set_xlabel("pathway rank")
        ax2.set_ylabel("fraction of samples with pathway")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 5: clinical duplicate handling + missingness ------------ #
        miss = clin.isna().mean(axis=0).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(11.69, 6))
        ax.hist(miss.values, bins=30, color="#C44E52")
        ax.set_title("Clinical variable missingness distribution")
        ax.set_xlabel("fraction missing")
        ax.set_ylabel("number of variables")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        dup_lines = ["Duplicate clinical codes (kept FIRST, dropped the rest):", ""]
        dup_lines += PROV.clinical_dup_notes or ["  (none)"]
        dup_lines += [
            "",
            "Samples in omics but WITHOUT usable clinical data (excluded):",
            "  " + (", ".join(PROV.dropped_no_clinical) or "(none)"),
        ]
        _text_page(pdf, "Clinical data quality & decisions", dup_lines)

    log.info("[report] done")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    log.info("=== multi-omics harmonization pipeline ===")

    # Load / convert each modality
    meta_g = load_metagenomics()
    clinical_full = convert_clinical()
    clinical = dedup_clinical(clinical_full)
    mapping = load_mapping()
    meta_b = load_metabolomics()

    # Harmonize to a common, identically-ordered sample set
    common, mg, mb, clin = harmonize(meta_g, meta_b, clinical, mapping)
    validate(common, mg, mb, clin)

    # Write deliverables
    mg.to_csv(OUT_METAGENOMICS, sep="\t")
    mb.to_csv(OUT_METABOLOMICS, sep="\t")
    clin.to_csv(OUT_CLINICAL, sep="\t")
    log.info("[write] %s", os.path.relpath(OUT_METAGENOMICS, ROOT))
    log.info("[write] %s", os.path.relpath(OUT_METABOLOMICS, ROOT))
    log.info("[write] %s", os.path.relpath(OUT_CLINICAL, ROOT))

    # Report
    build_report(common, mg, mb, clin)

    log.info(
        "=== done: %d harmonized samples (%d autism / %d control) ===",
        PROV.n_final,
        PROV.n_autism,
        PROV.n_control,
    )


if __name__ == "__main__":
    main()
