import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def normalize_icd9(code: Any) -> str:
    if pd.isna(code) or code == "":
        return ""
    text = str(code).upper().strip()
    normalized = re.sub(r"[^A-Z0-9]", "", text)
    normalized = normalized.lstrip("0")
    return normalized or text


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_patients(csv_dir: Path) -> pd.DataFrame:
    path = csv_dir / "PATIENTS.csv.gz"
    usecols = ["SUBJECT_ID", "GENDER", "DOB", "DOD", "EXPIRE_FLAG"]
    parse_dates = ["DOB", "DOD"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"SUBJECT_ID": "Int64", "GENDER": "category", "EXPIRE_FLAG": "Int64"},
        parse_dates=parse_dates,
        compression="gzip",
    )
    return df.rename(columns={"EXPIRE_FLAG": "EXPIRED"})


def load_admissions(csv_dir: Path) -> pd.DataFrame:
    path = csv_dir / "ADMISSIONS.csv.gz"
    usecols = [
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "ADMISSION_TYPE",
        "INSURANCE",
        "ETHNICITY",
        "RELIGION",
        "MARITAL_STATUS",
        "DIAGNOSIS",
        "HOSPITAL_EXPIRE_FLAG",
        "HAS_CHARTEVENTS_DATA",
    ]
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "SUBJECT_ID": "Int64",
            "HADM_ID": "Int64",
            "ADMISSION_TYPE": "category",
            "INSURANCE": "category",
            "ETHNICITY": "category",
            "RELIGION": "category",
            "MARITAL_STATUS": "category",
        },
        parse_dates=["ADMITTIME", "DISCHTIME"],
        compression="gzip",
    )
    return df


def load_diagnoses(csv_dir: Path) -> pd.DataFrame:
    path = csv_dir / "DIAGNOSES_ICD.csv.gz"
    df = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "ICD9_CODE": "string"},
        compression="gzip",
    )
    df["ICD9_CODE"] = df["ICD9_CODE"].str.strip()
    return df


def load_icd_dictionary(csv_dir: Path) -> Dict[str, str]:
    path = csv_dir / "D_ICD_DIAGNOSES.csv.gz"
    df = pd.read_csv(
        path,
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
        dtype={"ICD9_CODE": "string", "SHORT_TITLE": "string", "LONG_TITLE": "string"},
        compression="gzip",
    )
    df["ICD9_CODE"] = df["ICD9_CODE"].str.strip()
    df["Key"] = df["ICD9_CODE"].apply(normalize_icd9)
    df["Description"] = df["SHORT_TITLE"].fillna("")
    df.loc[df["Description"].str.len() < 3, "Description"] = df["LONG_TITLE"].fillna("")
    df["Description"] = df["Description"].str.replace("\n", " ", regex=False).str.strip()
    description_lookup = (
        df.dropna(subset=["Key"]).drop_duplicates("Key").set_index("Key")["Description"].to_dict()
    )
    return {k: v for k, v in description_lookup.items() if v}


def load_rare_codes(rare_file: Path) -> Set[str]:
    if not rare_file.exists():
        raise FileNotFoundError(f"Rare code file not found at {rare_file}")
    df = pd.read_csv(rare_file, dtype={"icd_9": "string"})
    return set(df["icd_9"].dropna().apply(normalize_icd9))


def build_demographic_summary(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    def summarize(title: str, series: pd.Series) -> pd.DataFrame:
        clean = series.copy()
        if pd.api.types.is_categorical_dtype(clean.dtype):
            clean = clean.cat.add_categories("Missing")
        clean = clean.fillna("Missing").astype(str)
        summary = clean.value_counts().reset_index()
        summary.columns = ["Value", "Count"]
        total = summary["Count"].sum()
        summary["Percent"] = (summary["Count"] / total * 100).round(1)
        summary.insert(0, "Attribute", title)
        return summary

    pieces = [
        summarize("Gender", patients["GENDER"]),
        summarize("Ethnicity", admissions["ETHNICITY"]),
        summarize("Admission Type", admissions["ADMISSION_TYPE"]),
        summarize("Insurance", admissions["INSURANCE"]),
        summarize("Religion", admissions["RELIGION"]),
        summarize("Marital Status", admissions["MARITAL_STATUS"]),
    ]
    return pd.concat(pieces, ignore_index=True)


def add_age_and_los(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    df = admissions.merge(patients[["SUBJECT_ID", "DOB", "GENDER"]], on="SUBJECT_ID", how="left")
    year_diff = df["ADMITTIME"].dt.year - df["DOB"].dt.year
    plausible = year_diff.abs() <= 200
    plausible &= df["ADMITTIME"].notna() & df["DOB"].notna()
    age = pd.Series(np.nan, index=df.index, dtype="Float64")
    if plausible.any():
        delta = (df.loc[plausible, "ADMITTIME"] - df.loc[plausible, "DOB"]).dt.total_seconds()
        age.loc[plausible] = delta / (365.25 * 24 * 3600)
    invalid = (~plausible) & df["ADMITTIME"].notna() & df["DOB"].notna()
    if invalid.any():
        LOGGER.warning("%d records have implausible admit/dob spans and will show missing age", invalid.sum())
    df["Age"] = age
    df["Age"] = df["Age"].clip(lower=0)
    df["LOS"] = (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / (24 * 60 * 60)
    df["LOS"] = df["LOS"].clip(lower=0)
    return df


def summarize_age(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    stats = (
        df.dropna(subset=["Age"])
        .groupby(group_col)["Age"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    stats = stats.rename(
        columns={"count": "Admissions", "mean": "MeanAge", "median": "MedianAge", "std": "StdDev"}
    )
    stats[["MeanAge", "MedianAge", "StdDev"]] = stats[["MeanAge", "MedianAge", "StdDev"]].round(1)
    return stats


def summarize_los(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    stats = (
        df.dropna(subset=["LOS"])
        .groupby(group_col)["LOS"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    stats = stats.rename(
        columns={"count": "Admissions", "mean": "MeanLOS", "median": "MedianLOS", "std": "StdDev"}
    )
    stats[["MeanLOS", "MedianLOS", "StdDev"]] = stats[["MeanLOS", "MedianLOS", "StdDev"]].round(1)
    return stats


def plot_histogram(series: pd.Series, path: Path, title: str, xlabel: str, bins=None, xlim=None) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(series.dropna(), bins=bins, color="#2c7fb8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Admissions")
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    LOGGER.info("Wrote %s", path)


def build_rare_disease_summary(
    diagnoses: pd.DataFrame,
    patients: pd.DataFrame,
    rare_set: Set[str],
    icd_lookup: Dict[str, str],
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnoses["CleanCode"] = diagnoses["ICD9_CODE"].apply(normalize_icd9)
    rare_df = diagnoses[diagnoses["CleanCode"].isin(rare_set)].copy()
    rare_df = rare_df.merge(patients[["SUBJECT_ID", "GENDER"]], on="SUBJECT_ID", how="left")
    if pd.api.types.is_categorical_dtype(rare_df["GENDER"].dtype):
        rare_df["GENDER"] = rare_df["GENDER"].cat.add_categories("Unknown")
    rare_df["GENDER"] = rare_df["GENDER"].fillna("Unknown")

    summary = (
        rare_df.groupby(["CleanCode", "ICD9_CODE"])
        .agg(Occurrences=("ICD9_CODE", "size"), Patients=("SUBJECT_ID", pd.Series.nunique))
        .reset_index()
    )
    summary["Description"] = summary["CleanCode"].map(icd_lookup).fillna("Description unavailable")
    summary["Label"] = summary.apply(lambda row: f"{row.ICD9_CODE} — {row.Description}", axis=1)
    summary = summary.sort_values(by="Patients", ascending=False).reset_index(drop=True)
    summary["Label"] = summary["Label"].str.replace("\n", " ", regex=False)
    top_codes = summary.head(top_n)

    gender_table = (
        rare_df[rare_df["CleanCode"].isin(top_codes["CleanCode"])]
        .groupby(["CleanCode", "GENDER"])
        .size()
        .reset_index(name="Count")
    )
    gender_table = gender_table.merge(
        summary[["CleanCode", "Label"]].drop_duplicates("CleanCode"), on="CleanCode", how="left"
    )
    gender_table["Label"] = pd.Categorical(
        gender_table["Label"], categories=summary.head(top_n)["Label"].tolist(), ordered=True
    )
    return summary, gender_table, rare_df


def _value_counts(series: pd.Series) -> pd.Series:
    clean = series.copy()
    if pd.api.types.is_categorical_dtype(clean.dtype):
        if "Missing" not in clean.cat.categories:
            clean = clean.cat.add_categories("Missing")
    clean = clean.fillna("Missing").astype(str)
    return clean.value_counts()


def _build_demographic_comparison(
    enriched: pd.DataFrame,
    rare: pd.DataFrame,
    columns: Tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for col in columns:
        all_counts = _value_counts(enriched[col])
        rare_counts = _value_counts(rare[col])
        categories = sorted(set(all_counts.index) | set(rare_counts.index))
        for cat in categories:
            total = all_counts.get(cat, 0)
            rare_val = rare_counts.get(cat, 0)
            percent = (rare_val / total * 100) if total else 0.0
            rows.append(
                {
                    "Metric": col,
                    "Category": cat,
                    "AllCount": total,
                    "RareCount": rare_val,
                    "RarePctOfAll": round(percent, 1) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_age_bucket_table(
    enriched: pd.DataFrame,
    rare: pd.DataFrame,
    bins: Tuple[int, ...],
    labels: Tuple[str, ...],
) -> pd.DataFrame:
    def bucket(series: pd.Series) -> pd.Series:
        return pd.cut(series.dropna(), bins=bins, labels=labels, include_lowest=True)

    all_bucket = bucket(enriched["Age"])
    rare_bucket = bucket(rare["Age"])
    categories = list(labels)
    rows = []
    for cat in categories:
        all_val = int((all_bucket == cat).sum())
        rare_val = int((rare_bucket == cat).sum())
        pct = (rare_val / all_val * 100) if all_val else 0.0
        rows.append(
            {
                "Metric": "AgeBucket",
                "Category": cat,
                "AllCount": all_val,
                "RareCount": rare_val,
                "RarePctOfAll": round(pct, 1) if all_val else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_summary_table(
    enriched: pd.DataFrame,
    rare: pd.DataFrame,
    diagnoses: pd.DataFrame,
    rare_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = []
    def safe_stats(series: pd.Series, func):
        vals = series.dropna()
        return float(func(vals)) if len(vals) else float("nan")

    summary.append(
        {
            "Metric": "Admissions",
            "All": len(enriched),
            "Rare": len(rare),
        }
    )
    summary.append(
        {
            "Metric": "Unique patients",
            "All": enriched["SUBJECT_ID"].nunique(),
            "Rare": rare["SUBJECT_ID"].nunique(),
        }
    )
    summary.append(
        {
            "Metric": "Distinct ICD-9 codes",
            "All": diagnoses["CleanCode"].nunique(),
            "Rare": rare_df["CleanCode"].nunique(),
        }
    )
    age_metrics = {
        "Mean Age": lambda vals: vals.mean(),
        "Median Age": lambda vals: vals.median(),
        "90th percentile age": lambda vals: vals.quantile(0.9),
    }
    for name, func in age_metrics.items():
        summary.append(
            {
                "Metric": name,
                "All": round(safe_stats(enriched["Age"], func), 1),
                "Rare": round(safe_stats(rare["Age"], func), 1),
            }
        )
    all_code_counts = diagnoses.groupby("HADM_ID")["ICD9_CODE"].nunique()
    rare_code_counts = rare_df.groupby("HADM_ID")["CleanCode"].nunique()
    summary.append(
        {
            "Metric": "Unique ICD-9 codes per admission (avg)",
            "All": round(all_code_counts.mean(), 2),
            "Rare": round(rare_code_counts.mean(), 2) if len(rare_code_counts) else 0.0,
        }
    )
    return pd.DataFrame(summary)


def _build_icd_prefix_table(diagnoses: pd.DataFrame, rare_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    def prefix(code: str) -> str:
        if not code:
            return "Missing"
        text = str(code)
        text = text.split(".")[0]
        return text[:3]

    diagnoses["Prefix"] = diagnoses["CleanCode"].apply(prefix)
    rare_df = rare_df.copy()
    rare_df["Prefix"] = rare_df["CleanCode"].apply(prefix)
    all_counts = diagnoses["Prefix"].value_counts().head(top_n)
    rare_counts = rare_df["Prefix"].value_counts()
    rows = []
    for prefix_val, count in all_counts.items():
        rare_val = rare_counts.get(prefix_val, 0)
        rows.append(
            {
                "Prefix": prefix_val,
                "AllCount": int(count),
                "RareCount": int(rare_val),
                "RarePctOfAll": round(rare_val / count * 100, 1) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_age_comparison(enriched: pd.DataFrame, rare: pd.DataFrame, path: Path) -> None:
    records = []
    for label, df in [("All", enriched), ("Rare", rare)]:
        records.append(pd.DataFrame({"Age": df["Age"].dropna(), "Group": label}))
    combined = pd.concat(records, ignore_index=True)
    plt.figure(figsize=(11, 6))
    sns.histplot(
        combined,
        x="Age",
        hue="Group",
        multiple="stack",
        binwidth=5,
        palette=["#1f77b4", "#ff7f0e"],
        shrink=0.9,
        edgecolor="white",
    )
    plt.title("Age distribution: all admissions vs. rare subset")
    plt.xlabel("Age (years)")
    plt.ylabel("Admissions")
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    LOGGER.info("Wrote %s", path)


def plot_gender_pies(enriched: pd.DataFrame, rare: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    palette = ["#2c7fb8", "#fdae61", "#7fc97f"]
    for ax, (label, df) in zip(axes, [("All admissions", enriched), ("Rare subset", rare)]):
        counts = _value_counts(df["GENDER"])
        counts = counts[counts > 0]
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%.1f%%",
            colors=palette[: len(counts)],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(label)
    plt.suptitle("Gender composition of all vs. rare admissions")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    LOGGER.info("Wrote %s", path)


def build_comparative_outputs(
    enriched: pd.DataFrame,
    rare: pd.DataFrame,
    diagnoses: pd.DataFrame,
    rare_df: pd.DataFrame,
    output_dir: Path,
    fig_dir: Path,
) -> None:
    age_bins = (0, 18, 30, 45, 60, 75, 90, 120)
    age_labels = ("0-17", "18-29", "30-44", "45-59", "60-74", "75-89", "90+")
    demo_columns = (
        "GENDER",
        "ETHNICITY",
        "ADMISSION_TYPE",
        "INSURANCE",
        "RELIGION",
        "MARITAL_STATUS",
    )
    demo_table = _build_demographic_comparison(enriched, rare, demo_columns)
    write_table(demo_table, output_dir / "rare_vs_all_demographics.csv")
    age_table = _build_age_bucket_table(enriched, rare, age_bins, age_labels)
    write_table(age_table, output_dir / "rare_vs_all_age_buckets.csv")
    summary_table = _build_summary_table(enriched, rare, diagnoses, rare_df)
    write_table(summary_table, output_dir / "rare_vs_all_summary.csv")
    prefix_table = _build_icd_prefix_table(diagnoses, rare_df, top_n=10)
    write_table(prefix_table, output_dir / "rare_vs_all_icd_prefixes.csv")
    plot_age_comparison(enriched, rare, fig_dir / "rare_vs_all_age.png")
    plot_gender_pies(enriched, rare, fig_dir / "rare_vs_all_gender.png")


def plot_top_rare_counts(summary: pd.DataFrame, path: Path, top_n: int) -> None:
    subset = summary.head(top_n).copy()
    subset = subset.iloc[::-1]
    plt.figure(figsize=(11, 6))
    sns.barplot(x="Patients", y="Label", data=subset, palette="mako")
    plt.title("Top rare diagnosis codes by unique patients")
    plt.xlabel("Unique patients")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    LOGGER.info("Wrote %s", path)


def plot_rare_gender_distribution(gender_table: pd.DataFrame, path: Path) -> None:
    order = (
        gender_table.drop_duplicates(subset=["CleanCode"]).set_index("CleanCode")["Label"].tolist()
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=gender_table,
        x="Count",
        y="Label",
        hue="GENDER",
        order=order,
        dodge=True,
    )
    plt.title("Rare diagnoses split by gender for the top codes")
    plt.xlabel("Occurrences")
    plt.ylabel("")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    LOGGER.info("Wrote %s", path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s (rows: %d)", path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset profiling for MIMIC-III CSV exports")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path(os.environ.get("MIMIC_CSV_DIR", "datasets/mimic3/csv")),
        help="Directory containing the MIMIC CSV exports (or set MIMIC_CSV_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dataset_analysis"),
        help="Where to write summary tables",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figs/dataset_analysis"),
        help="Where to write figures",
    )
    parser.add_argument(
        "--rare-file",
        type=Path,
        default=Path("datasets/mimic3_full/rare_icd9.csv"),
        help="CSV of rare ICD-9 codes (one per row)",
    )
    parser.add_argument(
        "--max-rare",
        type=int,
        default=8,
        help="How many rare codes to highlight in the figures",
    )
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    ensure_dir(args.output_dir)
    ensure_dir(args.fig_dir)
    sns.set_theme(style="whitegrid")

    patients = load_patients(args.csv_dir)
    admissions = load_admissions(args.csv_dir)
    enriched = add_age_and_los(admissions, patients)

    demo_summary = build_demographic_summary(patients, admissions)
    write_table(demo_summary, args.output_dir / "demographic_summary.csv")

    age_gender = summarize_age(enriched, "GENDER")
    age_ethnicity = summarize_age(enriched, "ETHNICITY")
    write_table(age_gender, args.output_dir / "age_by_gender.csv")
    write_table(age_ethnicity, args.output_dir / "age_by_ethnicity.csv")

    los_by_admission = summarize_los(enriched, "ADMISSION_TYPE")
    write_table(los_by_admission, args.output_dir / "los_by_admission_type.csv")

    plot_histogram(
        enriched["Age"],
        args.fig_dir / "age_distribution.png",
        "Age distribution at admission",
        "Age (years)",
        bins=np.arange(0, 121, 5),
        xlim=(0, 110),
    )
    plot_histogram(
        enriched["LOS"],
        args.fig_dir / "length_of_stay.png",
        "Length of stay distribution",
        "Length of stay (days)",
        bins=np.arange(0, 61, 1),
        xlim=(0, 60),
    )

    diagnoses = load_diagnoses(args.csv_dir)
    icd_lookup = load_icd_dictionary(args.csv_dir)
    rare_codes = load_rare_codes(args.rare_file)
    rare_summary, rare_gender, rare_df = build_rare_disease_summary(
        diagnoses, patients, rare_codes, icd_lookup, args.max_rare
    )
    write_table(rare_summary, args.output_dir / "rare_diseases.csv")
    plot_top_rare_counts(rare_summary, args.fig_dir / "top_rare_diseases.png", args.max_rare)
    plot_rare_gender_distribution(rare_gender, args.fig_dir / "rare_diseases_by_gender.png")

    rare_hadm_ids = rare_df["HADM_ID"].dropna().astype(str).unique()
    rare_admissions = enriched[enriched["HADM_ID"].astype(str).isin(rare_hadm_ids)]
    build_comparative_outputs(
        enriched,
        rare_admissions,
        diagnoses,
        rare_df,
        args.output_dir,
        args.fig_dir,
    )

    LOGGER.info("Dataset profiling finished. Tables in %s, figures in %s", args.output_dir, args.fig_dir)


if __name__ == "__main__":
    main()