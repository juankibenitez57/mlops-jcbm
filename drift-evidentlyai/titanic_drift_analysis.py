#!/usr/bin/env python3
"""
Titanic Data Drift Analysis — Evidently AI
==========================================
Genera 12 divisiones train/val/test del dataset Titanic bajo distintas
condiciones (estratificación, proporciones, semilla) y produce un informe
de deriva HTML por cada conjunto (val y test) usando EvidentlyAI.

Condiciones experimentales (2 × 3 × 2 = 12):
  - Estratificación: sí / no
  - Proporción de división: 60/20/20, 90/5/5, 98/1/1
  - Semilla aleatoria: 42, 123
"""

from __future__ import annotations

import json
import urllib.request
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/titanic")
REPORTS_DIR = Path("reports/titanic")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "titanic_raw.csv"
TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# ── Schema Titanic ────────────────────────────────────────────────────────────
TARGET_COL = "Survived"
NUMERICAL_COLS = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_COLS = ["Pclass", "Sex", "Embarked"]
ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Columnas sobre las que se calculará ValueDrift individualmente
DRIFT_COLS = ALL_FEATURE_COLS + [TARGET_COL]

titanic_schema = DataDefinition(
    numerical_columns=NUMERICAL_COLS,
    categorical_columns=CATEGORICAL_COLS + [TARGET_COL],
)

# ── Condiciones experimentales ────────────────────────────────────────────────
SEEDS = [42, 123]
SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),
    (0.90, 0.05, 0.05),
    (0.98, 0.01, 0.01),
]
STRATIFY_OPTIONS = [True, False]

CONDITIONS: list[dict] = [
    {
        "stratify": strat,
        "train_ratio": tr,
        "val_ratio": vr,
        "test_ratio": ter,
        "seed": seed,
    }
    for strat, (tr, vr, ter), seed in product(
        STRATIFY_OPTIONS, SPLIT_RATIOS, SEEDS
    )
]


# ── Funciones reutilizables ───────────────────────────────────────────────────

def load_titanic() -> pd.DataFrame:
    """Descarga el dataset si no existe y lo devuelve limpio."""
    if not RAW_CSV.exists():
        print(f"Descargando Titanic dataset → {RAW_CSV}")
        urllib.request.urlretrieve(TITANIC_URL, RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def split_dataset(
    df: pd.DataFrame, condition: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide df en train / val / test según la condición dada.

    Estrategia:
        1. Split train vs (val+test)
        2. Split (val+test) 50/50 → val y test
    """
    seed = condition["seed"]
    temp_ratio = condition["val_ratio"] + condition["test_ratio"]
    strat_col = df[TARGET_COL] if condition["stratify"] else None

    try:
        X_train, X_temp = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=seed,
            stratify=strat_col,
        )
    except ValueError:
        # Fallback sin estratificación si la muestra es demasiado pequeña
        X_train, X_temp = train_test_split(
            df, test_size=temp_ratio, random_state=seed
        )

    strat_temp = X_temp[TARGET_COL] if condition["stratify"] else None
    try:
        X_val, X_test = train_test_split(
            X_temp, test_size=0.5, random_state=seed, stratify=strat_temp
        )
    except ValueError:
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=seed)

    return X_train, X_val, X_test


def condition_label(cond: dict) -> str:
    """Etiqueta legible y trazable para una condición experimental."""
    strat = "strat-yes" if cond["stratify"] else "strat-no"
    tr = int(cond["train_ratio"] * 100)
    vr = int(cond["val_ratio"] * 100)
    ter = int(cond["test_ratio"] * 100)
    seed = cond["seed"]
    return f"{strat}_split-{tr}-{vr}-{ter}_seed-{seed}"


def make_evidently_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df.reset_index(drop=True), data_definition=titanic_schema
    )


def build_report() -> Report:
    """
    Informe con:
      - DataDriftPreset  → detección global + share de columnas con deriva
      - DataSummaryPreset → estadísticas descriptivas del dataset
      - ValueDrift por columna → p-valor y estado de deriva individual
    """
    per_col_metrics = [ValueDrift(column=col) for col in DRIFT_COLS]
    return Report(
        [DataDriftPreset(), DataSummaryPreset()] + per_col_metrics,
        include_tests=True,
    )


def extract_drift_stats(result) -> dict:
    """
    Extrae estadísticas de deriva del resultado de Evidently:
      - drift_share: fracción de columnas con deriva (DriftedColumnsCount)
      - drift_count: número de columnas con deriva
      - per_column: dict {col → bool} indicando si hay deriva por columna
    """
    stats: dict = {
        "drift_share": None,
        "drift_count": None,
        "per_column": {},
    }

    for v in result.metric_results.values():
        name = v.display_name

        # Métrica global de columnas con deriva
        if name == "Count of Drifted Columns":
            try:
                stats["drift_share"] = round(v.share.value, 4)
                stats["drift_count"] = int(v.count.value)
            except Exception:
                pass

        # Deriva por columna (v.tests son objetos MetricTestResult, no dicts)
        elif name.startswith("Value drift for "):
            col = name.removeprefix("Value drift for ")
            if col in DRIFT_COLS:
                try:
                    drifted = any(
                        getattr(getattr(t, "status", None), "value", "") == "FAIL"
                        for t in (v.tests or [])
                    )
                    stats["per_column"][col] = drifted
                except Exception:
                    stats["per_column"][col] = None

    return stats


# ── Ejecución principal ───────────────────────────────────────────────────────

def run_all() -> list[dict]:
    """
    Itera las 12 condiciones, genera informes HTML y devuelve
    una lista de registros con las estadísticas de deriva.
    """
    df = load_titanic()
    print(f"Dataset Titanic: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas usadas: {list(df.columns)}\n")

    records = []

    for i, cond in enumerate(CONDITIONS, 1):
        label = condition_label(cond)
        print(f"[{i:02d}/12] {label}")

        X_train, X_val, X_test = split_dataset(df, cond)

        print(
            f"        train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
        )

        ref_ds = make_evidently_dataset(X_train)

        for split_name, split_df in [("val", X_val), ("test", X_test)]:
            cur_ds = make_evidently_dataset(split_df)
            report = build_report()
            result = report.run(current_data=cur_ds, reference_data=ref_ds)

            html_path = REPORTS_DIR / f"{label}_{split_name}.html"
            result.save_html(str(html_path))

            stats = extract_drift_stats(result)
            records.append(
                {
                    "label": label,
                    "split": split_name,
                    "stratify": "Sí" if cond["stratify"] else "No",
                    "ratio": f"{int(cond['train_ratio']*100)}/{int(cond['val_ratio']*100)}/{int(cond['test_ratio']*100)}",
                    "seed": cond["seed"],
                    "n_train": len(X_train),
                    "n_split": len(split_df),
                    "drift_share": stats["drift_share"],
                    "drift_count": stats["drift_count"],
                    **{
                        f"drift_{col}": stats["per_column"].get(col)
                        for col in DRIFT_COLS
                    },
                }
            )
            drift_pct = (
                f"{stats['drift_share']*100:.0f}%"
                if stats["drift_share"] is not None
                else "N/A"
            )
            print(f"        [{split_name}] deriva={drift_pct} → {html_path.name}")

        print()

    return records


def print_summary_table(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    val_df = df[df["split"] == "val"][
        ["stratify", "ratio", "seed", "n_train", "n_split", "drift_share"]
    ].rename(columns={"n_split": "n_val", "drift_share": "drift_share_val"})
    test_df = df[df["split"] == "test"][
        ["stratify", "ratio", "seed", "drift_share"]
    ].rename(columns={"drift_share": "drift_share_test"})

    summary = val_df.reset_index(drop=True).copy()
    summary["drift_share_test"] = test_df["drift_share_test"].values

    summary.columns = [
        "Estratificación",
        "Ratio (tr/val/te)",
        "Semilla",
        "N train",
        "N val",
        "Deriva val (frac.)",
        "Deriva test (frac.)",
    ]

    print("\n" + "=" * 90)
    print("TABLA RESUMEN DE DERIVA")
    print("=" * 90)
    print(summary.to_string(index=True))
    print("=" * 90)


def save_stats_json(records: list[dict]) -> None:
    out = REPORTS_DIR / "drift_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nEstadísticas guardadas en {out}")


if __name__ == "__main__":
    records = run_all()
    print_summary_table(records)
    save_stats_json(records)
    print(
        f"\n[DONE] {len(records)} informes generados en '{REPORTS_DIR}/'.\n"
        "Consulta reports/titanic/drift_stats.json para los valores numéricos."
    )
